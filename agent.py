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

# Define the state for LangGraph workflow
class AnalysisState(TypedDict):
    question: str
    dataset_info: dict
    plan: str  # GPT-5-nano's analysis plan
    code: str  # Claude's generated code
    execution_result: str  # Tool execution result
    validation: str  # GPT-4o-mini's validation
    final_result: dict

#frist step: load data and profile the data
df = pd.read_csv('2025-06-25T01-21_export.csv')
profile_data = profile_dataframe_simple(df)  # 返回字典而不是JSON字符�?

# agent_tools.py


class DataAIAgent:
    def __init__(self, df: pd.DataFrame, openai_api_key=None, claude_api_key=None):
        self.df = df
        self.con = duckdb.connect()          
        self.con.register("t", df)           
        
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
            try:
                import numpy as np
                safe_globals = {"__builtins__": {}}
                safe_locals = {
                    "pd": pd,
                    "df": self.df.copy(),
                    "np": np,
                    "json": json,
                    "print": print
                }
                
                exec(code, safe_globals, safe_locals)
                result = safe_locals.get("result", "No result variable found")
                
                if isinstance(result, pd.DataFrame):
                    return f"DataFrame result with shape {result.shape}:\n{result.head(10).to_string()}"
                else:
                    return f"Result: {str(result)}"
                    
            except Exception as e:
                return f"Python Error: {str(e)}"
        
        def run_python_plot_tool(code: str) -> str:
            """Execute Python code for creating plots"""
            try:
                result = self.python_repl.run(code)
                
                # Check for generated files
                import glob
                png_files = glob.glob("*.png")
                if png_files:
                    latest_file = max(png_files, key=os.path.getctime)
                    return f"Plot created successfully: {latest_file}\nExecution result: {result}"
                else:
                    return f"Code executed but no plot file found. Make sure to use plt.savefig('filename.png'). Result: {result}"
                    
            except Exception as e:
                return f"Plot Error: {str(e)}"
        
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
    
    def plan_analysis(self, state: AnalysisState) -> AnalysisState:
        """Node 1: GPT-5-nano designs analysis plan"""
        print("🎯 GPT-5-nano: Designing analysis plan...")
        
        plan_prompt = f"""You are an expert data analyst. Given this question and dataset info, create a detailed analysis plan.

Question: {state['question']}

Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

Create a step-by-step analysis plan that includes:
1. What type of analysis is needed (calculation, visualization, etc.)
2. Which columns to use
3. What statistical methods or visualization types to apply
4. Expected output format

Respond in JSON format:
{{
    "analysis_type": "calculation|visualization|both",
    "columns_needed": ["col1", "col2"],
    "method": "describe the approach",
    "expected_output": "describe expected result"
}}"""

        try:
            response = self.gpt_llm1.invoke(plan_prompt)
            plan = response.content
            print(f"📋 Plan created: {plan[:200]}...")
            
            state["plan"] = plan
            return state
        except Exception as e:
            state["plan"] = f"Planning error: {str(e)}"
            return state
    
    def generate_code(self, state: AnalysisState) -> AnalysisState:
        """Node 2: Claude generates code based on plan"""
        print("💻 Claude: Generating code based on plan...")
        
        code_prompt = f"""You are a Python code generator. Based on this analysis plan, generate the appropriate code.

Analysis Plan: {state['plan']}
Question: {state['question']}
Dataset Info: {json.dumps(state['dataset_info'], ensure_ascii=False)}

Generate Python code that:
- Uses DataFrame 'df' for data analysis
- Assigns final result to 'result' variable
- For plots, use plt.savefig('analysis_chart.png')

Respond with just: ACTION_TYPE:CODE where ACTION_TYPE is either PYTHON or PLOT"""

        try:
            response = self.claude_llm.invoke(code_prompt)
            code_response = response.content
            print(f"🔧 Code generated: {code_response[:200]}...")
            
            state["code"] = code_response
            return state
        except Exception as e:
            state["code"] = f"Code generation error: {str(e)}"
            return state
    
    def execute_code(self, state: AnalysisState) -> AnalysisState:
        """Node 3: Execute the generated code using tools"""
        print("�?Executing generated code...")
        
        try:
            code_content = state["code"]
            
            if "PYTHON:" in code_content:
                code = code_content.split("PYTHON:")[1].strip()
                result = self.tools[0].func(code)  # run_python_tool
            elif "PLOT:" in code_content:
                code = code_content.split("PLOT:")[1].strip()
                result = self.tools[1].func(code)  # run_python_plot_tool
            else:
                result = f"Unknown code format: {code_content}"
            
            print(f"�?Execution result: {result[:200]}...")
            state["execution_result"] = result
            return state
            
        except Exception as e:
            state["execution_result"] = f"Execution error: {str(e)}"
            return state
    
    def validate_and_analyze(self, state: AnalysisState) -> AnalysisState:
        """Node 4: GPT-4o-mini validates code and analyzes results"""
        print("🛡�?GPT-4o-mini: Validating and analyzing results...")
        
        validation_prompt = f"""You are a code reviewer and data analyst. Review this analysis workflow:

Original Question: {state['question']}
Analysis Plan: {state['plan']}
Generated Code: {state['code']}
Execution Result: {state['execution_result']}

Provide comprehensive feedback in JSON format:
{{
    "code_quality": "assessment of code quality",
    "result_interpretation": "clear explanation of results",
    "correctness": "is the approach correct?",
    "recommendations": "suggestions for improvement",
    "final_answer": "direct answer to the original question"
}}"""

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
    
    def create_langgraph_workflow(self):
        """Create the LangGraph DAG workflow"""
        
        # Create the graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("plan", self.plan_analysis)
        workflow.add_node("generate", self.generate_code)
        workflow.add_node("execute", self.execute_code)
        workflow.add_node("validate", self.validate_and_analyze)
        
        # Define the DAG flow
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def langgraph_analysis(self, question: str, dataset_info: dict = None) -> dict:
        """Run the complete LangGraph DAG workflow"""
        
        print("=== LangGraph DAG Workflow ===")
        print("🔄 Flow: GPT-5-nano(plan) -> Claude(code) -> Tools(execute) -> GPT-4o-mini(validate)")
        
        # Initialize state
        initial_state = AnalysisState(
            question=question,
            dataset_info=dataset_info or profile_data,
            plan="",
            code="",
            execution_result="",
            validation="",
            final_result={}
        )
        
        # Create and run workflow
        workflow = self.create_langgraph_workflow()
        
        try:
            final_state = workflow.invoke(initial_state)
            return final_state["final_result"]
        except Exception as e:
            return {"error": f"Workflow error: {str(e)}"}



# Create LangChain-based data analysis agent
agent = DataAIAgent(df)

# Test LangGraph DAG workflow
if __name__ == "__main__":
    print("=== LangGraph DAG Data Agent Test ===")
    question = "Calculate the average '总价' by '名称'"
    
    # LangGraph DAG workflow (3-model collaboration)
    print("\n🚀 LangGraph DAG Workflow:")
    langgraph_result = agent.langgraph_analysis(question, profile_data)
    print("\n📊 Final Result:")
    print(json.dumps(langgraph_result, ensure_ascii=False, indent=2))


