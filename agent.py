import os
import json
import subprocess
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import pytz
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
load_dotenv()

# LangSmith Configuration (Optional)
# Automatically enabled if LANGCHAIN_TRACING_V2=true in .env
# No code changes needed - just set environment variables!
from function import profile_dataframe_json,profile_dataframe_simple
from function import profile_dataframe_simple_json
import json, duckdb, pandas as pd
from typing import Any, Dict
from openai import OpenAI
# LangChain imports
from langchain_experimental.tools import PythonREPLTool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema import SystemMessage
import tempfile
import base64

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Define the state for LangGraph 
class AnalysisState(TypedDict):
    question: str
    dataset_info: dict
    plan: str  # GPT-5-nano's analysis plan
    code: str  # Claude's generated code
    execution_result: str  # Tool execution result
    validation: str  # GPT-4o-mini's validation
    final_result: dict
    # Memory for multi-turn conversation
    conversation_history: List[Dict[str, str]]  # List of {role, content} messages
    plan_iterations: List[str]  # Track plan refinements
    user_feedback: str  # User feedback on current plan
    plan_confirmed: bool  # Whether user confirmed the plan

# agent_tools.py

class DataAIAgent:
    def __init__(self, df: pd.DataFrame, openai_api_key=None, claude_api_key=None):
        self.df = df
        self.con = duckdb.connect()          
        self.con.register("t", df)
        
        # Check LangSmith status
        if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
            project = os.getenv('LANGCHAIN_PROJECT', 'default')
            print(f"✅ LangSmith tracing enabled - Project: {project}")
            print(f"   View at: https://smith.langchain.com/")
        else:
            print("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
        
        # Initialize LangChain LLMs
        claude_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not claude_key:
            raise ValueError("Claude API key not found. Please set Claude_TOKEN environment variable or pass claude_api_key parameter.")
        
        self.claude_llm = ChatAnthropic(
            model="claude-sonnet-4-5",
            temperature=0,
            api_key=claude_key
        )
        
        self.gpt_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        self.gpt_llm1=ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        # Initialize Python REPL tool
        self.python_repl = PythonREPLTool()
        
        # Setup Python environment
        setup_code = """
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
plt.ioff()

df = pd.read_csv('2025-06-25T01-21_export.csv')
"""
        self.python_repl.run(setup_code)

        # Create LangChain tools for LangGraph workflow
        self.tools = self._create_langchain_tools()

    def _create_langchain_tools(self):
        """Create LangChain Tool objects for the agent"""
        
        def run_sql_tool(sql: str) -> str:
            """Execute SQL query on table 't'"""
            try:
                if any(keyword in sql.lower() for keyword in ["insert", "update", "delete", "drop"]):
                    return "Error: Only SELECT queries are allowed"
                
                result = self.con.execute(sql).df()
                preview = result.head(20)
                return f"SQL executed successfully. Shape: {result.shape}\nResults:\n{preview.to_string()}"
            except Exception as e:
                return f"SQL Error: {str(e)}"
        
        def run_python_tool(code: str) -> str:
            """Execute Python code for data analysis"""
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            try:
                # Create safe builtins whitelist
                safe_builtins = {
                    'len': len, 'range': range, 'sum': sum, 'min': min, 'max': max,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'zip': zip, 'enumerate': enumerate, 'sorted': sorted,
                    'abs': abs, 'round': round, 'int': int, 'float': float, 'str': str,
                    'bool': bool, 'any': any, 'all': all, 'map': map, 'filter': filter,
                    'isinstance': isinstance, 'type': type, 'print': print,
                    'True': True, 'False': False, 'None': None
                }
                
                # Prepare execution environment
                import numpy as np
                import scipy
                safe_globals = {"__builtins__": safe_builtins}
                safe_locals = {
                    "pd": pd,
                    "df": self.df.copy(),
                    "np": np,
                    "scipy": scipy,
                    "json": json
                }
                
                # Capture stdout and stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals, safe_locals)
                
                # Get captured output
                stdout_text = stdout_capture.getvalue()
                stderr_text = stderr_capture.getvalue()
                
                # Get result variable
                result = safe_locals.get("result", None)
                
                # Format response
                response_parts = []
                
                if stdout_text:
                    response_parts.append(f"Output:\n{stdout_text.strip()}")
                
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        response_parts.append(f"\nDataFrame result (shape {result.shape}):\n{result.head(10).to_string()}")
                    else:
                        response_parts.append(f"\nResult: {str(result)}")
                elif not stdout_text:
                    response_parts.append("Code executed successfully (no output or result variable)")
                
                if stderr_text:
                    response_parts.append(f"\nWarnings:\n{stderr_text.strip()}")
                
                return "\n".join(response_parts) if response_parts else "Code executed successfully"
                    
            except Exception as e:
                import traceback
                return f"Python Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        def run_python_plot_tool(code: str) -> str:
            """Execute Python code for creating plots"""
            import io
            import sys
            import glob
            import time
            from contextlib import redirect_stdout, redirect_stderr
            
            try:
                # Create safe builtins whitelist
                safe_builtins = {
                    'len': len, 'range': range, 'sum': sum, 'min': min, 'max': max,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'zip': zip, 'enumerate': enumerate, 'sorted': sorted,
                    'abs': abs, 'round': round, 'int': int, 'float': float, 'str': str,
                    'bool': bool, 'any': any, 'all': all, 'map': map, 'filter': filter,
                    'isinstance': isinstance, 'type': type, 'print': print,
                    'True': True, 'False': False, 'None': None
                }
                
                # Prepare execution environment with matplotlib
                import numpy as np
                import scipy
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.ioff()
                
                safe_globals = {"__builtins__": safe_builtins}
                safe_locals = {
                    "pd": pd,
                    "df": self.df.copy(),
                    "np": np,
                    "scipy": scipy,
                    "plt": plt,
                    "sns": sns,
                    "matplotlib": matplotlib,
                    "json": json
                }
                
                # Generate unique filename with timestamp
                timestamp = int(time.time() * 1000)
                unique_filename = f"plot_{timestamp}.png"
                
                # Get existing png files before execution
                existing_files = set(glob.glob("*.png"))
                
                # Set up beautiful plot styles before execution
                try:
                    # 1) Seaborn theme: clean white grid style
                    sns.set_theme(
                        style="whitegrid",      # whitegrid / white / darkgrid
                        context="talk",         # paper / notebook / talk / poster
                        font_scale=1.05,
                        rc={
                            "figure.dpi": 120,
                            "savefig.dpi": 160,
                            "figure.figsize": (7.5, 4.5),
                            "axes.spines.top": False,
                            "axes.spines.right": False,
                            "axes.titleweight": "semibold",
                            "axes.labelweight": "regular",
                            "axes.grid": True,
                            "grid.linestyle": "--",
                            "grid.alpha": 0.25,
                            "axes.linewidth": 1.0,
                            "lines.linewidth": 2,
                            "lines.markersize": 5,
                            "legend.frameon": False,
                            "legend.loc": "best",
                            "xtick.major.pad": 3,
                            "ytick.major.pad": 3,
                            "font.family": "DejaVu Sans",  # Or "Arial" / "Source Sans 3"
                        }
                    )
                    sns.set_palette("colorblind")  # Beautiful and readable colors
                except Exception:
                    pass
                
                # 2) Fallback matplotlib style (if not using seaborn)
                matplotlib.rcParams.update({
                    "figure.autolayout": True,     # Auto layout, less overlap
                    "axes.titlesize": "large",
                    "axes.labelsize": "medium",
                    "xtick.labelsize": "small",
                    "ytick.labelsize": "small",
                })
                
                # Capture stdout and stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals, safe_locals)
                
                # Close all matplotlib figures to free resources
                plt.close('all')
                
                # Get captured output
                stdout_text = stdout_capture.getvalue()
                stderr_text = stderr_capture.getvalue()
                
                # Check for newly generated files
                current_files = set(glob.glob("*.png"))
                new_files = current_files - existing_files
                
                # Format response
                response_parts = []
                
                if new_files:
                    files_list = ", ".join(sorted(new_files))
                    response_parts.append(f"✅ Plot created successfully: {files_list}")
                else:
                    response_parts.append("⚠️ Code executed but no plot file found. Make sure to use plt.savefig('filename.png')")
                
                if stdout_text:
                    response_parts.append(f"\nOutput:\n{stdout_text.strip()}")
                
                if stderr_text:
                    response_parts.append(f"\nWarnings:\n{stderr_text.strip()}")
                
                return "\n".join(response_parts)
                    
            except Exception as e:
                import traceback
                return f"Plot Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        # Create LangChain Tool objects
        tools = [
            Tool(
                name="run_sql",
                func=run_sql_tool,
                description="Execute SQL queries on table 't' to query and analyze data. Only SELECT queries are allowed."
            ),
            Tool(
                name="run_python", 
                func=run_python_tool,
                description="Execute Python code for data analysis using DataFrame 'df'. Assign final results to variable 'result'."
            ),
            Tool(
                name="run_python_plot",
                func=run_python_plot_tool, 
                description="Execute Python code to create visualizations and charts. Use plt.savefig('filename.png') to save plots."
            )
        ]
        
        return tools

    # ---- LangGraph DAG Workflow Nodes ----
    
    def interactive_planning(self, state: AnalysisState) -> AnalysisState:
        """Node 0: Interactive planning with memory - allows multi-turn conversation"""
        print("🤔 GPT-5-nano: Interactive planning session...")
        
        # Initialize conversation history if not exists
        if not state.get('conversation_history'):
            state['conversation_history'] = []
            state['plan_iterations'] = []
            state['user_feedback'] = ""
            state['plan_confirmed'] = False
        
        # Check if user confirmed the plan (via "confirm" command or explicit feedback)
        if state.get('user_feedback') and "confirm" in state.get('user_feedback', ''):
            state['plan_confirmed'] = True
            print("✅ User confirmed the plan, proceeding to execution...")
            return state
        
        # Build conversation context
        conversation_context = ""
        if state['conversation_history']:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in state['conversation_history'][-6:]:  # Keep last 6 messages for context
                conversation_context += f"{msg['role']}: {msg['content']}\n"
        
        # Build plan iterations context
        plan_context = ""
        if state['plan_iterations']:
            plan_context = "\n\nPrevious plan iterations:\n"
            for i, plan in enumerate(state['plan_iterations'][-3:], 1):  # Keep last 3 iterations
                plan_context += f"Iteration {i}: {plan}\n"
        
        planning_prompt = f"""You are a data analyst. Create an execution plan for the user's request.

Question: {state['question']}

Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

{conversation_context}

User Feedback: {state.get('user_feedback', 'None - First interaction')}

PRINCIPLE: 
- User asks for X → plan to deliver EXACTLY X
- Be concise and precise
- Example: "画一个直方图" → Plan: create histogram of [specific column]

Respond in JSON:
{{
    "analysis_plan": {{
        "analysis_type": "calculation|visualization|both",
        "columns_needed": ["col1", "col2"],
        "method": "Brief description of approach",
        "expected_output": "What will be delivered"
    }},
    "message_to_user": "I will [exact action]. Ready to execute?",
    "optional_suggestions": "Optional: 1 brief improvement idea (if truly valuable)"
}}

Keep all fields CONCISE. User will confirm before execution."""

        try:
            response = self.gpt_llm1.invoke(planning_prompt)
            planning_result = response.content
            
            # Add to conversation history
            state['conversation_history'].append({
                'role': 'assistant',
                'content': planning_result
            })
            
            # Try to parse the JSON response
            try:
                parsed_result = json.loads(planning_result)
                
                # Extract plan and status
                plan_str = json.dumps(parsed_result.get('analysis_plan', {}), ensure_ascii=False)
                state['plan_iterations'].append(plan_str)
                state['plan'] = plan_str
                
                # Store the full response for display
                state['planning_response'] = parsed_result
                
                # Always wait for user confirmation
                print(f"📋 {parsed_result.get('message_to_user', '')}")
                
                # Show suggestions if available
                if parsed_result.get('optional_suggestions'):
                    print(f"💡 Suggestions: {parsed_result.get('optional_suggestions')}")
                
                # Never auto-confirm, always wait for user
                state['plan_confirmed'] = False
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                state['plan'] = planning_result
                state['plan_iterations'].append(planning_result)
            
            return state
            
        except Exception as e:
            state['plan'] = f"Interactive planning error: {str(e)}"
            print(f"❌ Error in planning: {str(e)}")
            return state
    
    def generate_code(self, state: AnalysisState) -> AnalysisState:
        """Node 2: Claude generates code based on plan"""
        print("💻 Claude: Generating code based on plan...")
        
        # Parse the plan to determine analysis type
        plan_str = state.get('plan', '')
        analysis_type = "calculation"  # default
        
        try:
            if plan_str:
                plan_json = json.loads(plan_str)
                analysis_type = plan_json.get('analysis_type', 'calculation')
                print(f"   Analysis type: {analysis_type}")
        except:
            print("   Could not parse plan JSON, using default type: calculation")
        
        code_prompt = f"""You are a Python code generator. Based on this analysis plan, generate executable Python code.

Analysis Plan: {state['plan']}
Question: {state['question']}
Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

AVAILABLE ENVIRONMENT:
- Libraries: pandas (pd), numpy (np), scipy, matplotlib.pyplot (plt), seaborn (sns), json
- DataFrame 'df' is already loaded with the data
- Built-in functions: len, range, sum, min, max, list, dict, zip, enumerate, sorted, abs, round, etc.
- Note: plt and sns are only available in PLOT sections

IMPORTANT INSTRUCTIONS:
1. Generate ONLY raw Python code - NO markdown formatting, NO code blocks (```), NO explanations
2. For calculations: assign final result to variable 'result' and/or use print()
3. For Chinese column names, use: df["列名"] with double quotes
4. For plots: use plt.savefig('filename.png') to save the plot
5. Do NOT import libraries (they are already available)

Example format (calculation):
PYTHON:
result = df.groupby("类型")["数量"].sum()
print(result)

Example format (visualization):
PLOT:
plt.figure(figsize=(10, 6))
df["销量"].plot(kind="bar")
plt.title("Sales Analysis")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.tight_layout()
plt.savefig('sales_chart.png')


Now generate ONLY the raw Python code (no markdown, no explanations):"""

        try:
            print("   Calling Claude API...")
            response = self.claude_llm.invoke(code_prompt)
            code_response = response.content
            
            if not code_response or len(code_response.strip()) == 0:
                raise ValueError("Claude returned empty response")
            
            # Clean up the response: remove markdown code blocks if present
            code_response = code_response.strip()
            
            # Remove markdown code block markers
            if code_response.startswith("```"):
                lines = code_response.split('\n')
                # Remove first line (```python or similar)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove last line (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                code_response = '\n'.join(lines)
            
            # If still wrapped in backticks, try another approach
            code_response = code_response.replace("```python", "").replace("```", "").strip()
            
            print(f"🔧 Code generated successfully ({len(code_response)} chars)")
            print(f"   Preview: {code_response[:150]}...")
            
            state["code"] = code_response
            return state
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Code generation failed!")
            print(f"   Error: {str(e)}")
            print(f"   Details: {error_detail[:500]}")
            state["code"] = f"Code generation error: {str(e)}"
            return state
    
    def execute_code(self, state: AnalysisState) -> AnalysisState:
        """Node 3: Execute the generated code using tools"""
        print("⚡ Executing generated code...")
        
        try:
            code_content = state["code"]
            print(f"   Code length: {len(code_content)} chars")
            print(f"   Code preview: {code_content[:100]}...")
            results = []
            
            # Handle BOTH case: code might contain both PYTHON and PLOT sections
            if "PYTHON:" in code_content and "PLOT:" in code_content:
                print("📊 Detected both calculation and visualization tasks...")
                
                # Extract and execute Python code first
                python_part = code_content.split("PYTHON:")[1]
                if "PLOT:" in python_part:
                    python_code = python_part.split("PLOT:")[0].strip()
                else:
                    python_code = python_part.strip()
                
                print("  → Executing calculation code...")
                python_result = self.tools[1].func(python_code)  # run_python_tool
                results.append(f"Calculation Result:\n{python_result}")
                
                # Extract and execute Plot code
                plot_code = code_content.split("PLOT:")[1].strip()
                print("  → Executing visualization code...")
                plot_result = self.tools[2].func(plot_code)  # run_python_plot_tool
                results.append(f"\nVisualization Result:\n{plot_result}")
                
                result = "\n".join(results)
                
            elif "PYTHON:" in code_content:
                print("🔢 Executing calculation code...")
                code = code_content.split("PYTHON:")[1].strip()
                result = self.tools[1].func(code)  # run_python_tool
                
            elif "PLOT:" in code_content:
                print("📈 Executing visualization code...")
                code = code_content.split("PLOT:")[1].strip()
                result = self.tools[2].func(code)  # run_python_plot_tool
                
            else:
                # No explicit markers, try to detect if it's plotting code
                if any(keyword in code_content.lower() for keyword in ['plt.', 'plot', 'savefig', 'matplotlib']):
                    print("📈 Auto-detected visualization code...")
                    result = self.tools[2].func(code_content)  # run_python_plot_tool
                else:
                    print("🔢 Auto-detected calculation code...")
                    result = self.tools[1].func(code_content)  # run_python_tool
            
            print(f"✅ Execution result: {result[:200]}...")
            state["execution_result"] = result
            return state
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            state["execution_result"] = f"Execution error: {str(e)}\n\nDetails:\n{error_detail}"
            print(f"❌ Execution error: {str(e)}")
            return state
    
    def validate_and_analyze(self, state: AnalysisState) -> AnalysisState:
        """Node 4: GPT-4o-mini validates code and analyzes results"""
        print("🛡�?GPT-4o-mini: Validating and analyzing results...")
        
        validation_prompt = f"""You are an expert data analyst. Your task is to interpret the execution results and provide clear insights.

Original Question: {state['question']}
Execution Result: {state['execution_result']}
IMPORTANT: Focus on interpreting the results for the user. Provide clear, actionable insights.

Respond in JSON format:
{{
    "final_answer": "Direct, clear answer to the user's question based on the execution results",
    "result_interpretation": "Explain what the results mean in plain language. Highlight key findings, patterns, or insights",
    "recommendations": "Optional: suggestions for further analysis or improvements"
}}

Guidelines:
- final_answer: Should directly answer the user's question in 1-3 sentences
- result_interpretation: Explain the numbers/data in context, make it meaningful and detailed
- Keep it user-friendly
- If there are plots, mention what visualizations were created"""

        try:
            response = self.gpt_llm.invoke(validation_prompt)
            validation = response.content
            print(f"📊 Validation complete: {validation[:200]}...")
            
            state["validation"] = validation
            
            # Create final result
            state["final_result"] = {
                "question": state["question"],
                "plan": state["plan"],
                "code": state["code"],
                "execution_result": state["execution_result"],
                "validation": validation,
                "workflow": "GPT-5-nano(plan) -> Claude(code) -> Tools(execute) -> GPT-4o-mini(validate)"
            }
            
            return state
            
        except Exception as e:
            state["validation"] = f"Validation error: {str(e)}"
            state["final_result"] = {"error": str(e)}
            return state
    
    def add_user_feedback(self, state: AnalysisState, feedback: str) -> AnalysisState:
        """Add user feedback to the conversation and update state"""
        if not state.get('conversation_history'):
            state['conversation_history'] = []
        
        # Add user feedback to conversation history
        state['conversation_history'].append({
            'role': 'user',
            'content': feedback
        })
        
        state['user_feedback'] = feedback
        
        # Only reset confirmation if this is NOT a confirmation feedback
        # Check if feedback contains confirmation keywords
        is_confirmation = any(keyword in feedback.lower() for keyword in ['确认', 'confirm', '执行', 'execute', 'yes', '好的'])
        
        if not is_confirmation:
            state['plan_confirmed'] = False  # Reset confirmation when new feedback is added
        else:
            print(f"   ℹ️  Detected confirmation feedback, keeping plan_confirmed = {state.get('plan_confirmed', False)}")
        
        return state
    
    def should_continue_planning(self, state: AnalysisState) -> str:
        """Conditional edge function to determine if planning should continue"""
        if state.get('plan_confirmed', False):
            return "generate"
        else:
            return "interactive_plan"
    
    def create_langgraph_workflow(self):
        """Create the LangGraph DAG workflow with interactive planning"""
        
        # Create the graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("interactive_plan", self.interactive_planning)
        workflow.add_node("generate", self.generate_code)
        workflow.add_node("execute", self.execute_code)
        workflow.add_node("validate", self.validate_and_analyze)
        
        # Define the DAG flow with conditional routing
        workflow.set_entry_point("interactive_plan")
        
        # Conditional edge: continue planning or move to code generation
        workflow.add_conditional_edges(
            "interactive_plan",
            self.should_continue_planning,
            {
                "interactive_plan": "interactive_plan",  # Loop back for more planning
                "generate": "generate"  # Move to code generation when confirmed
            }
        )
        
        # Continue with normal flow after planning is confirmed
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def langgraph_analysis(self, question: str, dataset_info: dict = None, user_feedback: str = None, 
                          existing_state: AnalysisState = None) -> dict:
        """Run the complete LangGraph DAG workflow with interactive planning support"""
        
        print("=== Interactive LangGraph DAG Workflow ===")
        print("🔄 Flow: Interactive Planning -> Claude(code) -> Tools(execute) -> GPT-4o-mini(validate)")
        
        # Initialize or update state
        if existing_state:
            # Continue with existing conversation
            state = existing_state
            if user_feedback:
                state = self.add_user_feedback(state, user_feedback)
            
            print(f"📌 Debug in langgraph_analysis: plan_confirmed = {state.get('plan_confirmed', False)}")
            print(f"📌 Debug in langgraph_analysis: user_feedback = {state.get('user_feedback', 'None')[:50]}")
        else:
            # Initialize new state
            state = AnalysisState(
                question=question,
                dataset_info=dataset_info or profile_data,
                plan="",
                code="",
                execution_result="",
                validation="",
                final_result={},
                conversation_history=[],
                plan_iterations=[],
                user_feedback=user_feedback or "",
                plan_confirmed=False
            )
        
        try:
            # For interactive planning mode, just run one planning iteration
            # Check if plan is already confirmed
            if not state.get('plan_confirmed', False):
                print("🔄 Plan not confirmed yet, running interactive planning...")
                # Run only the interactive planning step
                state = self.interactive_planning(state)
                
                # Return the planning result for user review
                result = {
                    "question": state['question'],
                    "plan": state.get('plan', ''),
                    "code": "",
                    "execution_result": "",
                    "validation": "",
                    "conversation_state": {
                        "conversation_history": state.get("conversation_history", []),
                        "plan_iterations": state.get("plan_iterations", []),
                        "plan_confirmed": state.get("plan_confirmed", False),
                        "needs_user_input": not state.get("plan_confirmed", False)
                    }
                }
                return result
            
            # If plan is confirmed, run the full workflow
            else:
                print("✅ Plan already confirmed, skipping interactive planning and executing full workflow...")
                print(f"   Question: {state.get('question', 'N/A')[:50]}...")
                print(f"   Plan length: {len(state.get('plan', ''))} chars")
                
                # Step 1: Generate code
                print("\n🔹 Step 1: Generating code...")
                state = self.generate_code(state)
                if "error" in state.get("code", "").lower():
                    print(f"⚠️  Code generation had errors: {state['code'][:200]}")
                
                # Step 2: Execute code
                print("\n🔹 Step 2: Executing code...")
                state = self.execute_code(state)
                if "error" in state.get("execution_result", "").lower():
                    print(f"⚠️  Execution had errors: {state['execution_result'][:200]}")
                
                # Step 3: Validate and analyze
                print("\n🔹 Step 3: Validating results...")
                state = self.validate_and_analyze(state)
                
                # Return both the result and the state for continuation
                result = state.get("final_result", {})
                if not result:
                    # Construct result if not already set
                    result = {
                        "question": state['question'],
                        "plan": state.get('plan', ''),
                        "code": state.get('code', ''),
                        "execution_result": state.get('execution_result', ''),
                        "validation": state.get('validation', ''),
                        "workflow": "GPT-5-nano(plan) -> Claude(code) -> Tools(execute) -> GPT-4o-mini(validate)"
                    }
                
                result["conversation_state"] = {
                    "conversation_history": state.get("conversation_history", []),
                    "plan_iterations": state.get("plan_iterations", []),
                    "plan_confirmed": True,
                    "needs_user_input": False
                }
                
                print("\n✅ Workflow completed!")
                return result
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Error details:\n{error_detail}")
            return {"error": f"Workflow error: {str(e)}", "details": error_detail}
    
    def continue_conversation(self, existing_state: AnalysisState, user_feedback: str) -> dict:
        """Continue an existing conversation with new user feedback"""
        return self.langgraph_analysis(
            question=existing_state["question"],
            dataset_info=existing_state["dataset_info"],
            user_feedback=user_feedback,
            existing_state=existing_state
        )



# Interactive CLI for testing the agent
def interactive_cli(agent, profile_data):
    """Interactive CLI for testing the data analysis agent with memory"""
    print("="*70)
    print("🤖 AI Data Analysis Agent - Interactive CLI")
    print("="*70)
    print("\n📊 Dataset loaded successfully!")
    print(f"   Rows: {profile_data.get('dataset_info', {}).get('total_rows', 'N/A')}")
    print(f"   Columns: {profile_data.get('dataset_info', {}).get('total_columns', 'N/A')}")
    
    print("\n💡 Available columns:")
    for col in profile_data.get('column_overview', [])[:10]:
        col_name = col.get('column_name', '')
        samples = col.get('sample_values', [])
        sample_str = ', '.join([str(s) for s in samples[:2]])
        print(f"   • {col_name} (例如: {sample_str})")
    
    print("\n" + "="*70)
    print("Commands:")
    print("  - Type your question to start analysis")
    print("  - Type 'confirm' to confirm the plan and execute")
    print("  - Type 'new' to start a new analysis")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*70 + "\n")
    
    # Session state
    current_state = None
    current_question = None
    conversation_active = False
    
    while True:
        try:
            # Get user input
            if not conversation_active:
                user_input = input("🙋 Your question: ").strip()
            else:
                user_input = input("💬 Your feedback (or 'confirm' to proceed): ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'new':
                current_state = None
                current_question = None
                conversation_active = False
                print("\n🔄 Starting new analysis session...\n")
                continue
            
            if not user_input:
                print("⚠️  Please enter a question or command.")
                continue
            
            # Handle plan confirmation
            if user_input.lower() == 'confirm' and conversation_active and current_state:
                print("\n✅ Executing the confirmed plan...")
                print("⏳ Generating code and running analysis...\n")
                
                # Mark as confirmed and continue
                current_state['plan_confirmed'] = True
                current_state['user_feedback'] = "用户确认计划，请执行"
                
                result = agent.langgraph_analysis(
                    question=current_question,
                    dataset_info=profile_data,
                    user_feedback=current_state['user_feedback'],
                    existing_state=current_state
                )
                
                # Display results - clean and focused output
                print("\n" + "="*70)
                print("📊 ANALYSIS RESULTS")
                print("="*70)
                
                if result.get('error'):
                    print(f"\n❌ Error: {result['error']}")
                else:
                    print(f"\n📝 Question: {result.get('question', current_question)}")
                    
                    # Main output: Execution result (code output)
                    print(f"\n{'='*70}")
                    print("⚡ EXECUTION OUTPUT")
                    print(f"{'='*70}")
                    exec_result = result.get('execution_result', 'N/A')
                    print(exec_result)
                    
                    # Analysis interpretation from GPT-4o-mini
                    print(f"\n{'='*70}")
                    print("🔍 AI ANALYSIS & INTERPRETATION (GPT-4o-mini)")
                    print(f"{'='*70}")
                    validation = result.get('validation', 'N/A')
                    
                    # Try to parse and format validation JSON
                    try:
                        if validation and validation != 'N/A':
                            val_data = json.loads(validation)
                            
                            if val_data.get('final_answer'):
                                print(f"\n📌 Answer: {val_data['final_answer']}")
                            
                            if val_data.get('result_interpretation'):
                                print(f"\n💡 Interpretation:\n{val_data['result_interpretation']}")
                            
                            if val_data.get('code_quality'):
                                print(f"\n✓ Code Quality: {val_data['code_quality']}")
                            
                            if val_data.get('recommendations'):
                                print(f"\n💭 Recommendations:\n{val_data['recommendations']}")
                        else:
                            print(validation)
                    except json.JSONDecodeError:
                        print(validation)
                    
                    # Optional: show code and plan (can be hidden if not needed)
                    print(f"\n{'='*70}")
                    print("📋 DETAILS (Plan & Code)")
                    print(f"{'='*70}")
                    print(f"\nPlan: {result.get('plan', 'N/A')[:300]}...")
                    print(f"\nGenerated Code:\n{result.get('code', 'N/A')}")
                
                print("\n" + "="*70 + "\n")
                
                # Reset for new analysis
                conversation_active = False
                current_state = None
                current_question = None
                continue
            
            # Start new analysis or provide feedback
            if not conversation_active:
                # New question
                current_question = user_input
                print(f"\n🔍 Analyzing: {current_question}\n")
                print("⏳ GPT-5-nano is thinking...\n")
                
                result = agent.langgraph_analysis(current_question, profile_data)
                
                # Check if we need to store state for continuation
                conv_state = result.get('conversation_state', {})
                
                if conv_state.get('needs_user_input', False):
                    # Extract the planning response
                    conversation_active = True
                    
                    # Try to parse and display the plan (simplified)
                    try:
                        if result.get('plan'):
                            plan_data = json.loads(result['plan'])
                            
                            print("="*70)
                            print("📋 PROPOSED PLAN")
                            print("="*70)
                            
                            if isinstance(plan_data, dict):
                                analysis_plan = plan_data.get('analysis_plan', plan_data)
                                
                                # Show what will be done
                                if plan_data.get('message_to_user'):
                                    print(f"\n💬 {plan_data.get('message_to_user')}")
                                
                                # Show brief plan details
                                if analysis_plan:
                                    print(f"\n📊 Will use columns: {', '.join(analysis_plan.get('columns_needed', []))}")
                                    print(f"🎯 Type: {analysis_plan.get('analysis_type', 'N/A')}")
                                
                                # Show optional suggestions if available
                                if plan_data.get('optional_suggestions'):
                                    print(f"\n💡 Suggestion: {plan_data.get('optional_suggestions')}")
                        else:
                            # Display raw conversation history
                            print("="*70)
                            print("💭 CONVERSATION")
                            print("="*70)
                            for msg in conv_state.get('conversation_history', [])[-2:]:
                                role = "🤖 Assistant" if msg['role'] == 'assistant' else "👤 You"
                                print(f"\n{role}: {msg['content'][:500]}")
                    
                    except json.JSONDecodeError:
                        print(f"\n📋 Plan: {result.get('plan', 'N/A')[:500]}")
                    
                    print("\n" + "="*70)
                    print("\n💡 You can:")
                    print("   • Provide feedback to refine the plan")
                    print("   • Type 'confirm' to execute this plan")
                    print("   • Type 'new' to start over")
                    print("")
                    
                    # Store state for continuation
                    current_state = {
                        'question': current_question,
                        'dataset_info': profile_data,
                        'plan': result.get('plan', ''),
                        'code': '',
                        'execution_result': '',
                        'validation': '',
                        'final_result': {},
                        'conversation_history': conv_state.get('conversation_history', []),
                        'plan_iterations': conv_state.get('plan_iterations', []),
                        'user_feedback': '',
                        'plan_confirmed': False
                    }
                    
                else:
                    # Plan was auto-confirmed, show results
                    print("\n" + "="*70)
                    print("📊 ANALYSIS COMPLETE")
                    print("="*70)
                    print(f"\n📝 Question: {current_question}")
                    print(f"\n📋 Plan:\n{result.get('plan', 'N/A')[:500]}")
                    print(f"\n💻 Code:\n{result.get('code', 'N/A')[:500]}")
                    print(f"\n⚡ Result:\n{result.get('execution_result', 'N/A')[:500]}")
                    print("\n" + "="*70 + "\n")
                    
                    conversation_active = False
                    current_state = None
            
            else:
                # Continuing conversation with feedback
                print(f"\n💬 Feedback: {user_input}")
                print("⏳ Refining the plan...\n")
                
                result = agent.langgraph_analysis(
                    question=current_question,
                    dataset_info=profile_data,
                    user_feedback=user_input,
                    existing_state=current_state
                )
                
                # Update state
                conv_state = result.get('conversation_state', {})
                current_state['conversation_history'] = conv_state.get('conversation_history', [])
                current_state['plan_iterations'] = conv_state.get('plan_iterations', [])
                current_state['plan'] = result.get('plan', current_state.get('plan', ''))
                
                # Display updated plan
                try:
                    if result.get('plan'):
                        plan_data = json.loads(result['plan'])
                        
                        print("="*70)
                        print("📋 UPDATED ANALYSIS PLAN")
                        print("="*70)
                        
                        if isinstance(plan_data, dict):
                            analysis_plan = plan_data.get('analysis_plan', plan_data)
                            
                            print(f"\n🎯 Analysis Type: {analysis_plan.get('analysis_type', 'N/A')}")
                            print(f"\n📊 Columns: {', '.join(analysis_plan.get('columns_needed', []))}")
                            print(f"\n🔧 Method: {analysis_plan.get('method', 'N/A')}")
                            print(f"\n📈 Output: {analysis_plan.get('expected_output', 'N/A')}")
                            
                            if plan_data.get('message_to_user'):
                                print(f"\n💬 {plan_data.get('message_to_user')}")
                except:
                    print(f"\n📋 Updated Plan:\n{result.get('plan', 'N/A')[:500]}")
                
                print("\n" + "="*70)
                print("\n💡 Type 'confirm' to execute, or provide more feedback\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'new' to start over.\n")


# Main entry point
if __name__ == "__main__":
    # Load test data (only when running directly, not when imported)
    df = pd.read_csv('2025-06-25T01-21_export.csv')
    profile_data = profile_dataframe_simple(df)
    agent = DataAIAgent(df)
    
    # Run interactive CLI
    interactive_cli(agent, profile_data)


