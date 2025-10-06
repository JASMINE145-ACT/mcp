import os
import json
import duckdb
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
# LangSmith Configuration (Optional)
# Automatically enabled if LANGCHAIN_TRACING_V2=true in .env
# No code changes needed - just set environment variables!
from function import profile_dataframe_simple
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from config import require_secret
# Try to import Gemini (optional)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

openai_api_key=require_secret("OPENAI_API_KEY")
gemini_api_key=require_secret("GOOGLE_API_KEY")
claude_api_key=require_secret("ANTHROPIC_API_KEY")
# Define the state for LangGraph 
class AnalysisState(TypedDict):
    question: str
    dataset_info: dict
    plan: str  # GPT-5-nano's analysis plan
    code: str  # Claude's generated code
    code_validation: str  # Gemini's code logic validation
    code_approved: bool  # Whether code passed Gemini validation
    execution_result: str  # Tool execution result
    validation: str  # GPT-4o-mini's validation
    final_result: dict
    # Memory for multi-turn conversation
    conversation_history: List[Dict[str, str]]  # List of {role, content} messages
    plan_iterations: List[str]  # Track plan refinements
    user_feedback: str  # User feedback on current plan
    plan_confirmed: bool  # Whether user confirmed the plan
    # Error handling and retry
    execution_error: bool  # Whether execution had errors
    error_message: str  # Detailed error message for retry
    retry_count: int  # Number of retry attempts

# agent_tools.py

class DataAIAgent:
    def __init__(self, df: pd.DataFrame, openai_api_key=None, claude_api_key=None, gemini_api_key=None):
        self.df = df
        self.con = duckdb.connect()          
        self.con.register("t", df)
        
        # Initialize matplotlib styles once (performance optimization)
        self._setup_plot_styles()
        
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
            max_tokens=2000,  # Limit output to encourage concise code
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
        
        # Initialize Gemini for code validation (optional)
        self.gemini_llm = None
        if GEMINI_AVAILABLE:
            gemini_key = gemini_api_key or os.getenv('GOOGLE_API_KEY')
            if gemini_key:
                try:
                    self.gemini_llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-exp",  # Using Gemini 2.0 Flash (2.5 Pro not yet available in API)
                        temperature=0,
                        api_key=gemini_key
                    )
                    print("✅ Gemini model initialized for code validation")
                except Exception as e:
                    print(f"⚠️  Gemini initialization failed: {e}")
                    print("   Code validation will be skipped.")
            else:
                print("⚠️  Gemini API key not found. Code validation will be skipped.")
        else:
            print("ℹ️  Gemini not installed. Code validation will be skipped.")
            print("   (Optional: pip install langchain-google-genai google-generativeai)")
        
        # Create LangChain tools for LangGraph workflow
        self.tools = self._create_langchain_tools()
        
        # Create tool mapping by name for O(1) lookup
        # Model can directly use tool names from self.tools
        self.tool_map = {tool.name: tool for tool in self.tools}
    
    def _setup_plot_styles(self):
        """Initialize matplotlib/seaborn styles once for better performance"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            import platform
            
            matplotlib.use('Agg')
            plt.ioff()
            
            # Configure Chinese font support
            system = platform.system()
            if system == 'Windows':
                # Windows: Use Microsoft YaHei or SimHei
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
            elif system == 'Darwin':  # macOS
                plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
            else:  # Linux
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
            
            # Fix minus sign display issue in Chinese fonts
            plt.rcParams['axes.unicode_minus'] = False
            
            # Seaborn theme: clean white grid style
            sns.set_theme(
                style="whitegrid",
                context="talk",
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
                }
            )
            sns.set_palette("colorblind")
            
            # Fallback matplotlib style
            matplotlib.rcParams.update({
                "figure.autolayout": True,
                "axes.titlesize": "large",
                "axes.labelsize": "medium",
                "xtick.labelsize": "small",
                "ytick.labelsize": "small",
            })
            
            print("✅ 中文字体配置成功")
        except Exception as e:
            print(f"⚠️  Warning: Could not set up plot styles: {e}")

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
            """Unified Python execution tool: handles both data analysis and visualization"""
            import io
            import sys
            import glob
            import time
            import os
            from contextlib import redirect_stdout, redirect_stderr
            
            try:
                # Get project root directory
                project_root = os.path.dirname(os.path.abspath(__file__))
                
                # Prepare execution environment with matplotlib
                import numpy as np
                import scipy
                import sklearn
                import matplotlib
                import platform
                import statsmodels
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.ioff()
                
                # Configure Chinese font support for this execution
                system = platform.system()
                if system == 'Windows':
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
                elif system == 'Darwin':  # macOS
                    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
                else:  # Linux
                    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
                plt.rcParams['axes.unicode_minus'] = False
                
                # Change working directory to project root for saving plots
                original_cwd = os.getcwd()
                os.chdir(project_root)
                
                # Use restricted globals but allow necessary imports
                safe_globals = {
                    "__builtins__": {
                        'len': len, 'range': range, 'sum': sum, 'min': min, 'max': max,
                        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                        'zip': zip, 'enumerate': enumerate, 'sorted': sorted,
                        'abs': abs, 'round': round, 'int': int, 'float': float, 'str': str,
                        'bool': bool, 'any': any, 'all': all, 'map': map, 'filter': filter,
                        'isinstance': isinstance, 'type': type, 'print': print,
                        'True': True, 'False': False, 'None': None,
                        '__import__': __import__,  # Allow imports
                        'ImportError': ImportError,
                        'Exception': Exception,
                    }
                }
                
                safe_locals = {
                    "pd": pd,
                    "df": self.df.copy(),  # Copy needed to prevent modifications to original
                    "np": np,
                    "scipy": scipy,
                    "sklearn": sklearn,
                    "plt": plt,
                    "sns": sns,
                    "matplotlib": matplotlib,
                    "json": json,
                    "os": os
                }
                
                # Get existing png files in project root before execution
                existing_files = set(glob.glob(os.path.join(project_root, "*.png")))
                
                # Capture stdout and stderr (styles already set in __init__)
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        exec(code, safe_globals, safe_locals)
                    
                    # Close all matplotlib figures to free resources
                    plt.close('all')
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)
                
                # Get captured output
                stdout_text = stdout_capture.getvalue()
                stderr_text = stderr_capture.getvalue()
                
                # Check for newly generated files in project root
                current_files = set(glob.glob(os.path.join(project_root, "*.png")))
                new_files = current_files - existing_files
                # Extract just the filenames
                new_files = {os.path.basename(f) for f in new_files}
                
                # Get result variable (for calculations)
                result = safe_locals.get("result", None)
                
                # Format response - smart formatting based on what was produced
                response_parts = []
                
                # Add calculation results if any
                if stdout_text:
                    response_parts.append(f"Output:\n{stdout_text.strip()}")
                
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        response_parts.append(f"\nDataFrame result (shape {result.shape}):\n{result.head(10).to_string()}")
                    else:
                        response_parts.append(f"\nResult: {str(result)}")
                
                # Add plot results if any
                if new_files:
                    files_list = ", ".join(sorted(new_files))
                    response_parts.append(f"\n✅ Plot created successfully: {files_list}")
                
                # Handle case where code executed but produced nothing
                if not response_parts:
                    response_parts.append("Code executed successfully")
                
                if stderr_text:
                    response_parts.append(f"\nWarnings:\n{stderr_text.strip()}")
                
                return "\n".join(response_parts)
                    
            except Exception as e:
                import traceback
                return f"Python Execution Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
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
                description="Execute Python code for data analysis and visualization. Can handle calculations, statistics, and plots. Use 'result' variable for calculations, use plt.savefig('filename.png') for plots. Supports pandas, numpy, scipy, sklearn (scikit-learn), matplotlib, and seaborn."
            )
        ]
        
        return tools

    # ---- LangGraph DAG Workflow Nodes ----
    
    def _get_available_tools_description(self) -> str:
        """Generate dynamic tool list description for prompts"""
        tool_desc = "AVAILABLE TOOLS:\n"
        for tool_name, tool in self.tool_map.items():
            tool_desc += f'- "{tool_name}": {tool.description}\n'
        return tool_desc
    
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
        
        planning_prompt = f"""You are a data analyst assistant having a conversation with the user.

Question: {state['question']}
Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}
{conversation_context}
User Feedback: {state.get('user_feedback', 'None - First interaction')}

YOUR TASK:
1. If the user's request is CLEAR and SPECIFIC (e.g., "画一个直方图", "计算平均价格"):
   → Respond with a JSON plan (ready to execute)

2. If the user's request is VAGUE or EXPLORATORY (e.g., "这个数据怎么分析?", "有什么发现?"):
   → Respond with natural conversation text to clarify requirements
   → Ask questions, suggest options, or provide insights
   → Continue dialogue until the goal is clear

RESPONSE FORMATS:

A) For CLEAR requests - JSON format:
{{
    "analysis_plan": {{
        "analysis_type": "calculation|visualization|both",
        "columns_needed": ["col1", "col2"],
        "method": "Step by step approach with chain of thought",
        "expected_output": "What will be delivered"
    }},
    "message_to_user": "I will [exact action]. Ready to execute?",
    "optional_suggestions": "Optional: Brief improvement ideas"
}}

B) For VAGUE requests or ongoing conversation - Plain text:
Just respond naturally in conversational text. Examples:
- "我看了一下数据，有几个分析方向：1) 价格分布分析 2) 时间趋势分析 3) 分类对比。你想看哪个？"
- "要分析价格的话，我可以帮你：计算统计指标、画分布图、或者做分组对比。你更关心哪方面？"
- "这个数据集包含X列Y行，主要字段有...，你想探索什么问题？"

IMPORTANT:
- Keep conversation natural and helpful
- Only produce JSON when you have a CONCRETE, EXECUTABLE plan
- Method should show step-by-step reasoning
- Be concise but informative"""

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
                # Try to extract JSON from markdown code blocks if present
                json_text = planning_result.strip()
                if json_text.startswith("```"):
                    lines = json_text.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    json_text = '\n'.join(lines).strip()
                
                parsed_result = json.loads(json_text)
                
                # Check if it's a valid plan JSON (has analysis_plan key)
                if 'analysis_plan' in parsed_result:
                    # This is a concrete plan - extract it
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
                    
                    # Mark that we have a plan but need confirmation
                    state['plan_confirmed'] = False
                else:
                    # JSON but not a plan - treat as conversation
                    raise json.JSONDecodeError("Not a plan JSON", json_text, 0)
                
            except json.JSONDecodeError:
                # This is conversational text, not a plan
                print(f"💬 Assistant: {planning_result}")
                
                # Store as conversation but not as a plan
                state['plan'] = ""  # No concrete plan yet
                state['plan_confirmed'] = False
                
                # Mark that we're still in conversation mode
                state['planning_response'] = {
                    'conversation_mode': True,
                    'message': planning_result
                }
            
            return state
            
        except Exception as e:
            state['plan'] = f"Interactive planning error: {str(e)}"
            print(f"❌ Error in planning: {str(e)}")
            return state
    
    def generate_code(self, state: AnalysisState) -> AnalysisState:
        """Node 2: Claude generates code based on plan"""
        retry_count = state.get('retry_count', 0)
        
        if retry_count > 0:
            print(f"🔄 Claude: Regenerating code (Retry {retry_count}/3)...")
        else:
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
        
        # Choose prompt based on whether this is a retry
        if state.get('execution_error', False) and state.get('error_message'):
            # Retry scenario: Use dedicated error-fixing prompt
            print("   Using error-fixing prompt...")
            code_prompt = f"""You are a Python code debugger. The previous code execution failed. Your task is to analyze the error and generate CORRECTED code.

Question: {state['question']}
Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

PREVIOUS CODE (FAILED):
{state.get('code', 'N/A')}

ERROR MESSAGE:
{state['error_message']}

AVAILABLE ENVIRONMENT:
- Libraries: pandas (pd), numpy (np), scipy, sklearn (scikit-learn), matplotlib.pyplot (plt), seaborn (sns), json
- DataFrame 'df' is already loaded with the data
- Built-in functions: len, range, sum, min, max, list, dict, zip, enumerate, sorted, abs, round, etc.
- The unified Python tool can handle both calculations AND plots in the same code block

MACHINE LEARNING LIBRARY CHOICE:
- **Prefer sklearn** for classification/regression - it's more robust and handles data types automatically
- Only use statsmodels if specifically needed for statistical inference (p-values, significance tests)

CRITICAL INSTRUCTIONS:
1. Analyze the error message carefully
2. Fix ONLY the specific error - do not change working parts
3. Pay attention to:
   - Exact column names from dataset (case-sensitive)
   - Data types and operations compatibility
   - Missing values or empty data handling

{self._get_available_tools_description()}

OUTPUT FORMAT - Start with "PYTHON:" marker:
PYTHON:
# Your corrected code here
# Can include both calculations and plots
# For calculations: use print() or assign to 'result'
# For plots: use plt.savefig('filename.png')

Example:
PYTHON:
result = df['correct_column'].describe()
print(result)

Generate the CORRECTED code now (start with PYTHON:):"""

        else:
            # Initial generation: Use simple text-based format
            print("   Using initial code generation prompt...")
            code_prompt = f"""You are a Python code generator. Based on this analysis plan, generate executable Python code.

Analysis Plan: {state['plan']}
Question: {state['question']}
Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

AVAILABLE ENVIRONMENT:
- Libraries: pandas (pd), numpy (np), scipy, sklearn (scikit-learn), matplotlib.pyplot (plt), seaborn (sns), json, os
- DataFrame 'df' is already loaded with the data
- Built-in functions: len, range, sum, min, max, list, dict, zip, enumerate, sorted, abs, round, etc.
- The unified Python tool can handle both calculations AND plots in the same code block

MACHINE LEARNING LIBRARY CHOICE:
- **For classification/regression tasks**: Use sklearn (scikit-learn) - it's more robust and handles data types automatically
  Example: from sklearn.linear_model import LogisticRegression, LinearRegression
- **Only use statsmodels** if the user explicitly asks for statistical inference (p-values, confidence intervals, significance tests)
  Example: import statsmodels.api as sm

IMPORTANT INSTRUCTIONS:
1. For calculations: assign final result to variable 'result' and/or use print()
2. For Chinese column names, use: df["列名"] with double quotes
3. For plots: use plt.savefig('filename.png') to save the plot
4. Do NOT import libraries (they are already available)
5. Pay careful attention to column names - they must EXACTLY match the dataset
6. **KEEP CODE CONCISE** - Generate simple, direct code without comments or steps


{self._get_available_tools_description()}

OUTPUT FORMAT :
# Your code here - can include both calculations and plots
# For calculations: use print() or assign to 'result'
# For plots: use plt.savefig('filename.png')

Example output (combined calculation and plot):
# Analysis
result = df.groupby('category')['value'].agg(['mean', 'count'])
print(result)
# Visualization
plt.figure(figsize=(10, 6))
df.boxplot(column='value', by='category')
plt.savefig('boxplot.png')

Now generate CONCISE code:"""

        try:
            print("   Calling Claude API...")
            # Simple system message emphasizing conciseness
            from langchain.schema import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content="You are a Python code generator. Generate CONCISE, executable Python code starting with 'PYTHON:' marker. The unified tool handles both calculations and plots in the same block. Keep code short and direct - avoid verbose comments and unnecessary steps."),
                HumanMessage(content=code_prompt)
            ]
            response = self.claude_llm.invoke(messages)
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
            
            # Remove any remaining backticks
            code_response = code_response.replace("```python", "").replace("```", "").strip()
            
            print(f"🔧 Code generated successfully ({len(code_response)} chars)")
            print(f"   Preview: {code_response[:100]}...")
            
            # Store the text-based code directly
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
    
    def validate_code_logic(self, state: AnalysisState) -> AnalysisState:
        """Node 2.5: Gemini validates code logic before execution"""
        
        # Skip validation if Gemini is not available
        if not self.gemini_llm:
            print("⚠️  Gemini not available, skipping code validation...")
            state["code_validation"] = "Skipped: Gemini API key not configured"
            state["code_approved"] = True
            return state
        
        print("🔍 Gemini: Validating code logic and feasibility...")
        
        validation_prompt = f"""You are a Python code reviewer specializing in logic validation. Your task is to validate if the generated code:
1. Is logically correct and will execute without errors
2. Properly addresses the analysis plan requirements
3. Uses correct column names and data operations
4. Has proper error handling for edge cases

ANALYSIS PLAN:
{state['plan']}

GENERATED CODE:
{state['code']}

Respond in JSON format:
{{
    "approved": true/false,
    "confidence_score": 0-100,
    "issues_found": [
        {{"severity": "critical/warning/info", "description": "issue description", "suggestion": "how to fix"}}
    ],
    "logic_assessment": "Brief assessment of whether code logic matches plan requirements",
}}

IMPORTANT:
- Mark as approved:false only if there are CRITICAL issues that will cause execution failure
- For minor improvements, mark as approved:true but list them as warnings
- """

        try:
            response = self.gemini_llm.invoke(validation_prompt)
            validation_result = response.content
            
            print(f"📋 Validation result received ({len(validation_result)} chars)")
            
            # Try to parse JSON response
            try:
                # Try to extract JSON from markdown code blocks if present
                json_text = validation_result.strip()
                
                # Remove markdown code block markers if present
                if json_text.startswith("```"):
                    lines = json_text.split('\n')
                    # Remove first line (```json or similar)
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove last line (```)
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    json_text = '\n'.join(lines).strip()
                
                validation_data = json.loads(json_text)
                
                approved = validation_data.get('approved', True)
                confidence = validation_data.get('confidence_score', 0)
                issues = validation_data.get('issues_found', [])
                
                state["code_validation"] = validation_result
                state["code_approved"] = approved
                
                # Display validation summary
                if approved:
                    print(f"✅ Code approved (confidence: {confidence}%)")
                else:
                    print(f"❌ Code validation failed (confidence: {confidence}%)")
                
                # Display critical issues
                critical_issues = [i for i in issues if i.get('severity') == 'critical']
                if critical_issues:
                    print(f"   ⚠️  {len(critical_issues)} critical issue(s) found:")
                    for issue in critical_issues[:3]:  # Show first 3
                        print(f"      • {issue.get('description', '')}")
                        if issue.get('suggestion'):
                            print(f"        → {issue.get('suggestion', '')}")
                
                # Display warnings
                warnings = [i for i in issues if i.get('severity') == 'warning']
                if warnings:
                    print(f"   💡 {len(warnings)} warning(s) found")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Could not parse Gemini validation response ({str(e)}), defaulting to approved")
                state["code_validation"] = validation_result
                state["code_approved"] = True
            
            return state
            
        except Exception as e:
            print(f"❌ Gemini validation error: {str(e)}")
            # On error, default to approved to not block workflow
            state["code_validation"] = f"Validation error: {str(e)}"
            state["code_approved"] = True
            return state
    
    def execute_code(self, state: AnalysisState) -> AnalysisState:
        """Node 3: Execute the generated code using structured routing"""
        print("⚡ Executing generated code (structured routing)...")
        
        try:
            code_content = state["code"]
            results = []
            
            # Try to parse as structured JSON
            try:
                code_json = json.loads(code_content)
                tasks = code_json.get("tasks", [])
                
                if not tasks:
                    raise ValueError("No tasks found in JSON")
                
                print(f"📋 Found {len(tasks)} task(s) to execute")
                
                # Execute each task in order using O(1) routing with tool_map
                for i, task in enumerate(tasks, 1):
                    tool_name = task.get("tool")
                    code = task.get("code", "")
                    
                    print(f"\n  → Task {i}/{len(tasks)}: {tool_name}")
                    print(f"     Code preview: {code[:80]}...")
                    
                    # O(1) routing using tool_map
                    if tool_name in self.tool_map:
                        tool = self.tool_map[tool_name]
                        result = tool.func(code)
                        results.append(f"Task {i} ({tool_name}):\n{result}")
                    else:
                        available_tools = ', '.join(self.tool_map.keys())
                        results.append(f"Task {i}: Unknown tool '{tool_name}'. Available: {available_tools}")
                        print(f"     ⚠️  Unknown tool: {tool_name}")
                
                result = "\n\n".join(results)
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: try string-based parsing with PYTHON: or PLOT: markers
                print(f"⚠️  JSON parsing failed ({str(e)}), falling back to text marker parsing...")
                
                # Handle PYTHON: marker (new unified format)
                if "PYTHON:" in code_content:
                    print("🐍 Detected PYTHON: marker...")
                    code = code_content.split("PYTHON:")[1].strip()
                    # Remove PLOT: marker if it exists (legacy format)
                    if "PLOT:" in code:
                        # Combine both parts since run_python now handles both
                        python_part = code.split("PLOT:")[0].strip()
                        plot_part = code.split("PLOT:")[1].strip()
                        code = python_part + "\n\n" + plot_part
                    result = self.tool_map["run_python"].func(code)
                
                # Handle legacy PLOT: only marker
                elif "PLOT:" in code_content:
                    print("📊 Detected legacy PLOT: marker...")
                    code = code_content.split("PLOT:")[1].strip()
                    result = self.tool_map["run_python"].func(code)
                
                # No markers - execute directly with unified Python tool
                else:
                    print("⚡ No markers found, executing as Python code...")
                    result = self.tool_map["run_python"].func(code_content)
            
            # Check if execution had errors
            error_keywords = ["error:", "error ", "exception:", "traceback:", "failed", "valueerror", "typeerror", "keyerror", "attributeerror", "indexerror"]
            has_error = any(keyword in result.lower() for keyword in error_keywords)
            
            if has_error:
                print(f"❌ Execution encountered errors!")
                state["execution_error"] = True
                state["error_message"] = result
                state["execution_result"] = result
            else:
                print(f"✅ Execution successful")
                state["execution_error"] = False
                state["error_message"] = ""
                state["execution_result"] = result
            
            return state
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = f"Execution error: {str(e)}\n\nDetails:\n{error_detail}"
            state["execution_result"] = error_msg
            state["execution_error"] = True
            state["error_message"] = error_msg
            print(f"❌ Execution error: {str(e)}")
            return state
    
    def explain_results(self, state: AnalysisState) -> AnalysisState:
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
            workflow_desc = "GPT-5-nano(plan) -> Claude(code)"
            if self.gemini_llm:
                workflow_desc += " -> Gemini(validate)"
            workflow_desc += " -> Tools(execute) -> GPT-4o-mini(analyze)"
            
            state["final_result"] = {
                "question": state["question"],
                "plan": state["plan"],
                "code": state["code"],
                "code_validation": state.get("code_validation", "N/A"),
                "execution_result": state["execution_result"],
                "validation": validation,
                "workflow": workflow_desc
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
    
    def should_validate_or_execute(self, state: AnalysisState) -> str:
        """Conditional edge function after code generation: validate with Gemini or skip to execution"""
        # If Gemini is available, validate first
        if self.gemini_llm:
            return "validate"
        else:
            # Skip validation if Gemini not available
            return "execute"
    
    def should_execute_after_validation(self, state: AnalysisState) -> str:
        """Conditional edge function after Gemini validation"""
        approved = state.get('code_approved', True)
        retry_count = state.get('retry_count', 0)
        max_retries = 3
        
        if not approved and retry_count < max_retries:
            # Code failed validation, regenerate
            print(f"⚠️  Code validation failed, regenerating... (Attempt {retry_count + 1}/{max_retries})")
            state['retry_count'] = retry_count + 1
            # Use validation feedback as error message for regeneration
            state['error_message'] = f"Code validation failed:\n{state.get('code_validation', '')}"
            state['execution_error'] = True  # Treat validation failure as execution error for retry
            return "generate"
        elif not approved and retry_count >= max_retries:
            # Max retries reached, proceed anyway with warning
            print(f"⚠️  Max retries ({max_retries}) reached, executing despite validation concerns...")
            return "execute"
        else:
            # Code approved, proceed to execution
            return "execute"
    
    def should_retry_execution(self, state: AnalysisState) -> str:
        """Conditional edge function to determine if code execution should be retried"""
        # Check if there was an execution error
        has_error = state.get('execution_error', False)
        retry_count = state.get('retry_count', 0)
        max_retries = 3
        
        if has_error and retry_count < max_retries:
            # Increment retry count and retry
            print(f"⚠️  Execution failed, retrying... (Attempt {retry_count + 1}/{max_retries})")
            state['retry_count'] = retry_count + 1
            return "generate"  # Go back to code generation with error feedback
        elif has_error and retry_count >= max_retries:
            # Max retries reached, proceed to explain with error
            print(f"❌ Max retries ({max_retries}) reached, proceeding with error explanation...")
            return "explain"
        else:
            # Success, proceed to explanation
            print("✅ Execution successful, proceeding to explanation...")
            return "explain"
    
    def create_langgraph_workflow(self):
        """Create the LangGraph DAG workflow with interactive planning, Gemini validation, and error retry"""
        
        # Create the graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("interactive_plan", self.interactive_planning)
        workflow.add_node("generate", self.generate_code)
        workflow.add_node("validate", self.validate_code_logic)  # New Gemini validation node
        workflow.add_node("execute", self.execute_code)
        workflow.add_node("explain", self.explain_results)
        
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
        
        # After code generation, conditionally validate with Gemini or skip to execution
        workflow.add_conditional_edges(
            "generate",
            self.should_validate_or_execute,
            {
                "validate": "validate",  # Validate with Gemini if available
                "execute": "execute"  # Skip validation if Gemini not available
            }
        )
        
        # After validation, decide whether to regenerate or execute
        workflow.add_conditional_edges(
            "validate",
            self.should_execute_after_validation,
            {
                "generate": "generate",  # Regenerate if validation failed
                "execute": "execute"  # Execute if approved
            }
        )
        
        # Conditional edge after execution: retry if error, otherwise proceed to explain
        workflow.add_conditional_edges(
            "execute",
            self.should_retry_execution,
            {
                "generate": "generate",  # Retry code generation if execution failed
                "explain": "explain"  # Proceed to explanation if successful or max retries reached
            }
        )
        
        # End workflow after explanation
        workflow.add_edge("explain", END)
        
        return workflow.compile()
    
    def langgraph_analysis(self, question: str, dataset_info: dict = None, user_feedback: str = None, 
                          existing_state: AnalysisState = None) -> dict:
        """Run the complete LangGraph DAG workflow with interactive planning support"""
        
        print("=== Interactive LangGraph DAG Workflow ===")
        if self.gemini_llm:
            print("🔄 Flow: Interactive Planning -> Claude(code) -> Gemini(validate) -> Tools(execute) -> GPT-4o-mini(explain)")
        else:
            print("🔄 Flow: Interactive Planning -> Claude(code) -> Tools(execute) -> GPT-4o-mini(explain)")
        
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
                code_validation="",
                code_approved=True,
                execution_result="",
                validation="",
                final_result={},
                conversation_history=[],
                plan_iterations=[],
                user_feedback=user_feedback or "",
                plan_confirmed=False,
                execution_error=False,
                error_message="",
                retry_count=0
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
                
                # Step 2: Validate code logic with Gemini (if available)
                if self.gemini_llm:
                    print("\n🔹 Step 2: Validating code logic with Gemini...")
                    state = self.validate_code_logic(state)
                    
                    # If validation failed, regenerate code
                    retry_count = 0
                    max_retries = 3
                    while not state.get('code_approved', True) and retry_count < max_retries:
                        retry_count += 1
                        print(f"\n🔄 Regenerating code based on validation feedback (Attempt {retry_count}/{max_retries})...")
                        state['retry_count'] = retry_count
                        state['error_message'] = f"Code validation failed:\n{state.get('code_validation', '')}"
                        state['execution_error'] = True
                        
                        state = self.generate_code(state)
                        state = self.validate_code_logic(state)
                    
                    if not state.get('code_approved', True):
                        print(f"⚠️  Max validation retries reached, proceeding with execution anyway...")
                
                # Step 3: Execute code
                print("\n🔹 Step 3: Executing code...")
                state = self.execute_code(state)
                if "error" in state.get("execution_result", "").lower():
                    print(f"⚠️  Execution had errors: {state['execution_result'][:200]}")
                
                # Step 4: Validate and analyze results
                print("\n🔹 Step 4: Analyzing results with GPT-4o-mini...")
                state = self.explain_results(state)
                
                # Return both the result and the state for continuation
                result = state.get("final_result", {})
                if not result:
                    # Construct result if not already set
                    workflow_desc = "GPT-5-nano(plan) -> Claude(code)"
                    if self.gemini_llm:
                        workflow_desc += " -> Gemini(validate)"
                    workflow_desc += " -> Tools(execute) -> GPT-4o-mini(analyze)"
                    
                    result = {
                        "question": state['question'],
                        "plan": state.get('plan', ''),
                        "code": state.get('code', ''),
                        "code_validation": state.get('code_validation', 'N/A'),
                        "execution_result": state.get('execution_result', ''),
                        "validation": state.get('validation', ''),
                        "workflow": workflow_desc
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
                # Check if we have a concrete plan to confirm
                if not current_state.get('plan') or current_state.get('plan') == "":
                    print("⚠️  没有具体的执行计划可以确认。请先回答助手的问题，明确你的需求。\n")
                    continue
                
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
                    
                    # Check if we have a concrete plan or just conversation
                    has_concrete_plan = result.get('plan') and result.get('plan') != ""
                    
                    if has_concrete_plan:
                        # We have a concrete plan - show it nicely
                        try:
                            plan_data = json.loads(result['plan'])
                            
                            print("="*70)
                            print("📋 PROPOSED PLAN")
                            print("="*70)
                            
                            if isinstance(plan_data, dict):
                                analysis_plan = plan_data.get('analysis_plan', plan_data)
                                
                                # Show brief plan details
                                if analysis_plan:
                                    print(f"\n📊 Columns: {', '.join(analysis_plan.get('columns_needed', []))}")
                                    print(f"🎯 Type: {analysis_plan.get('analysis_type', 'N/A')}")
                                    print(f"🔧 Method: {analysis_plan.get('method', 'N/A')}")
                        
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Pure conversation mode - assistant is asking for clarification
                        # The message was already printed in interactive_planning
                        pass
                    
                    print("\n" + "="*70)
                    if has_concrete_plan:
                        print("\n💡 You can:")
                        print("   • Type 'confirm' to execute this plan")
                        print("   • Provide feedback to refine the plan")
                        print("   • Type 'new' to start over")
                    else:
                        print("\n💡 Please respond:")
                        print("   • Answer the question or clarify your needs")
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


