"""
Chainlit Application for AI Data Analysis Agent
基于 agent.py 的完整 Chainlit 实现
"""

import chainlit as cl
import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import from local modules
from agent import DataAIAgent
from function import profile_dataframe_simple

# Configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

@cl.on_chat_start
async def start():
    """初始化聊天会话"""
    await cl.Message(
        content="# 🤖 AI 数据分析助手\n\n"
                "欢迎使用智能数据分析系统！\n\n"
                "**功能特色：**\n"
                "- 🔍 智能对话式分析\n"
                "- 📊 自动生成可视化\n"
                "- 🧠 多模型协作（GPT-5-nano + Claude + Gemini + GPT-4o-mini）\n"
                "- 💬 多轮交互式规划\n\n"
                "**开始使用：**\n"
                "请上传 CSV 文件开始分析"
    ).send()
    
    # Request file upload
    files = await cl.AskFileMessage(
        content="📁 请上传您的数据文件（支持 CSV 或 Excel）",
        accept=["text/csv", ".csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx", ".xls"],
        max_size_mb=100,
        timeout=180,
    ).send()
    
    if files:
        file = files[0]
        await process_uploaded_file(file)

async def process_uploaded_file(file):
    """处理上传的文件"""
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get file path (Chainlit already saves the file)
        file_path = Path(file.path)
        file_name = file.name
        
        await msg.stream_token("📊 正在加载数据...")
        
        # Load data based on file extension
        file_ext = file_path.suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            await cl.Message(
                content=f"❌ 不支持的文件格式：{file_ext}\n\n请上传 CSV 或 Excel 文件。"
            ).send()
            return
        
        await msg.stream_token(f"\n✅ 数据加载成功！\n\n")
        await msg.stream_token(f"**数据概览：**\n")
        await msg.stream_token(f"- 文件名：{file_name}\n")
        await msg.stream_token(f"- 行数：{len(df):,}\n")
        await msg.stream_token(f"- 列数：{len(df.columns)}\n")
        
        # Get file size
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb < 1:
            await msg.stream_token(f"- 文件大小：{file_size_bytes / 1024:.2f} KB\n\n")
        else:
            await msg.stream_token(f"- 文件大小：{file_size_mb:.2f} MB\n\n")
        
        # Profile data
        profile_data = profile_dataframe_simple(df)
        
        await msg.stream_token("**可用列：**\n")
        for col in profile_data.get('column_overview', [])[:10]:
            col_name = col.get('column_name', '')
            samples = col.get('sample_values', [])
            sample_str = ', '.join([str(s) for s in samples[:2]])
            await msg.stream_token(f"- `{col_name}` (例如: {sample_str})\n")
        
        # Initialize agent
        agent = DataAIAgent(df)
        
        # Store in session
        cl.user_session.set("agent", agent)
        cl.user_session.set("profile_data", profile_data)
        cl.user_session.set("df", df)
        cl.user_session.set("file_name", file.name)
        cl.user_session.set("conversation_state", None)
        
        await msg.stream_token("\n\n✨ **准备就绪！** 您可以开始提问了。\n\n")
        await msg.stream_token("💡 **示例问题：**\n")
        await msg.stream_token("- 这个数据集有什么特点？\n")
        await msg.stream_token("- 帮我分析一下价格分布\n")
        await msg.stream_token("- 画一个箱线图按类别分组\n")
        
        await msg.update()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ 文件处理失败：{str(e)}\n\n请检查文件格式是否正确。"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """处理用户消息"""
    agent = cl.user_session.get("agent")
    profile_data = cl.user_session.get("profile_data")
    conversation_state = cl.user_session.get("conversation_state")
    
    if not agent:
        await cl.Message(
            content="⚠️ 请先上传 CSV 文件！\n\n使用 `/upload` 命令重新上传文件。"
        ).send()
        return
    
    # Handle commands
    user_input = message.content.strip()
    
    # Check for confirm command
    if user_input.lower() in ['confirm', '确认', '执行']:
        await handle_confirm(agent, profile_data, conversation_state)
        return
    
    # Check for new session command
    if user_input.lower() in ['new', 'restart', '重新开始']:
        cl.user_session.set("conversation_state", None)
        await cl.Message(content="🔄 已重置会话，可以开始新的分析了！").send()
        return
    
    # Process regular question
    await process_question(agent, profile_data, conversation_state, user_input)

async def process_question(agent, profile_data, conversation_state, question):
    """处理用户问题"""
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Show thinking indicator
        await msg.stream_token("🤔 **GPT-5-nano 正在思考...**\n\n")
        
        # Run analysis
        result = agent.langgraph_analysis(
            question=question,
            dataset_info=profile_data,
            user_feedback=None,
            existing_state=conversation_state
        )
        
        conv_state = result.get('conversation_state', {})
        
        # Check if we need user input (planning phase)
        if conv_state.get('needs_user_input', False):
            await handle_planning_response(msg, result, conv_state)
        else:
            # Execution completed
            await handle_execution_result(msg, result)
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        await cl.Message(
            content=f"❌ **发生错误：**\n\n```\n{str(e)}\n```\n\n"
                    f"<details><summary>详细信息</summary>\n\n```\n{error_detail}\n```\n</details>"
        ).send()

async def handle_planning_response(msg, result, conv_state):
    """处理规划阶段的响应"""
    has_concrete_plan = result.get('plan') and result.get('plan') != ""
    
    if has_concrete_plan:
        # We have a concrete plan
        await msg.stream_token("📋 **分析计划已生成**\n\n")
        
        try:
            plan_data = json.loads(result['plan'])
            analysis_plan = plan_data
            
            await msg.stream_token(f"**分析类型：** {analysis_plan.get('analysis_type', 'N/A')}\n\n")
            
            cols = analysis_plan.get('columns_needed', [])
            if cols:
                await msg.stream_token(f"**使用列：** {', '.join([f'`{c}`' for c in cols])}\n\n")
            
            method = analysis_plan.get('method', '')
            if method:
                await msg.stream_token(f"**方法：**\n{method}\n\n")
            
            expected = analysis_plan.get('expected_output', '')
            if expected:
                await msg.stream_token(f"**预期输出：** {expected}\n\n")
            
        except json.JSONDecodeError:
            await msg.stream_token(f"```json\n{result.get('plan', '')}\n```\n\n")
        
        # Store state for confirmation
        cl.user_session.set("conversation_state", {
            'question': result.get('question'),
            'dataset_info': cl.user_session.get("profile_data"),
            'plan': result.get('plan', ''),
            'code': '',
            'code_validation': '',
            'code_approved': True,
            'execution_result': '',
            'validation': '',
            'final_result': {},
            'conversation_history': conv_state.get('conversation_history', []),
            'plan_iterations': conv_state.get('plan_iterations', []),
            'user_feedback': '',
            'plan_confirmed': False,
            'execution_error': False,
            'error_message': '',
            'retry_count': 0
        })
        
        # Add action buttons
        actions = [
            cl.Action(name="confirm", value="confirm", label="✅ 确认执行"),
            cl.Action(name="modify", value="modify", label="✏️ 修改计划"),
        ]
        await msg.stream_token("\n\n---\n\n")
        await msg.stream_token("💡 **下一步：**\n")
        await msg.stream_token("- 点击 **✅ 确认执行** 开始分析\n")
        await msg.stream_token("- 或提供反馈修改计划\n")
        
    else:
        # Conversational mode - agent is asking for clarification
        # The message is already in conversation history
        last_msg = conv_state.get('conversation_history', [])[-1] if conv_state.get('conversation_history') else {}
        
        if last_msg and last_msg.get('role') == 'assistant':
            await msg.stream_token(f"💬 {last_msg.get('content', '')}\n\n")
        
        # Store minimal state
        cl.user_session.set("conversation_state", {
            'question': result.get('question'),
            'dataset_info': cl.user_session.get("profile_data"),
            'plan': '',
            'code': '',
            'code_validation': '',
            'code_approved': True,
            'execution_result': '',
            'validation': '',
            'final_result': {},
            'conversation_history': conv_state.get('conversation_history', []),
            'plan_iterations': [],
            'user_feedback': '',
            'plan_confirmed': False,
            'execution_error': False,
            'error_message': '',
            'retry_count': 0
        })
        
        await msg.stream_token("---\n\n")
        await msg.stream_token("💡 请回答上述问题以继续...\n")

async def handle_confirm(agent, profile_data, conversation_state):
    """处理确认执行"""
    if not conversation_state or not conversation_state.get('plan'):
        await cl.Message(
            content="⚠️ 没有可执行的计划。请先提出一个具体的分析问题。"
        ).send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        await msg.stream_token("✅ **开始执行分析**\n\n")
        await msg.stream_token("---\n\n")
        
        # Mark as confirmed
        conversation_state['plan_confirmed'] = True
        conversation_state['user_feedback'] = "用户确认计划，请执行"
        
        # Step 1: Code Generation
        await msg.stream_token("### 🔹 步骤 1: 生成代码\n")
        await msg.stream_token("💻 Claude 正在生成代码...\n\n")
        
        result = agent.langgraph_analysis(
            question=conversation_state['question'],
            dataset_info=profile_data,
            user_feedback=conversation_state['user_feedback'],
            existing_state=conversation_state
        )
        
        if result.get('code'):
            code_preview = result['code'][:200] + "..." if len(result['code']) > 200 else result['code']
            await msg.stream_token(f"✅ 代码已生成\n\n")
            await msg.stream_token(f"<details><summary>查看生成的代码</summary>\n\n```python\n{result.get('code', '')}\n```\n</details>\n\n")
        
        # Step 2: Validation (if Gemini available)
        if result.get('code_validation') and result.get('code_validation') != "N/A":
            await msg.stream_token("### 🔹 步骤 2: 代码验证\n")
            await msg.stream_token("🔍 Gemini 正在验证代码逻辑...\n\n")
            
            try:
                validation_data = json.loads(result['code_validation'])
                if validation_data.get('approved'):
                    confidence = validation_data.get('confidence_score', 0)
                    await msg.stream_token(f"✅ 代码通过验证 (置信度: {confidence}%)\n\n")
                else:
                    await msg.stream_token(f"⚠️ 代码存在问题，正在重新生成...\n\n")
            except:
                await msg.stream_token(f"✅ 验证完成\n\n")
        
        # Step 3: Execution
        await msg.stream_token("### 🔹 步骤 3: 执行分析\n")
        await msg.stream_token("⚡ 正在运行代码...\n\n")
        
        exec_result = result.get('execution_result', '')
        
        # Check for plots
        import re
        plot_pattern = r'Plot created successfully: ([\w\-\.]+\.png)'
        plots = re.findall(plot_pattern, exec_result)
        
        if plots:
            await msg.stream_token("✅ 分析完成！\n\n")
            await msg.stream_token("**执行结果：**\n")
            # Remove plot messages from output
            clean_output = re.sub(plot_pattern, '', exec_result)
            await msg.stream_token(f"```\n{clean_output.strip()}\n```\n\n")
            
            # Display plots
            await msg.stream_token("**📊 生成的图表：**\n\n")
            for plot_file in plots:
                plot_path = Path(plot_file)
                if plot_path.exists():
                    # Send image
                    image = cl.Image(path=str(plot_path), name=plot_file, display="inline")
                    await cl.Message(
                        content=f"**{plot_file}**",
                        elements=[image]
                    ).send()
        else:
            await msg.stream_token("✅ 分析完成！\n\n")
            await msg.stream_token("**执行结果：**\n")
            await msg.stream_token(f"```\n{exec_result}\n```\n\n")
        
        # Step 4: AI Analysis
        await msg.stream_token("### 🔹 步骤 4: AI 分析与解读\n")
        await msg.stream_token("🧠 GPT-4o-mini 正在分析结果...\n\n")
        
        validation = result.get('validation', '')
        try:
            val_data = json.loads(validation)
            
            if val_data.get('final_answer'):
                await msg.stream_token(f"**📌 答案：**\n{val_data['final_answer']}\n\n")
            
            if val_data.get('result_interpretation'):
                await msg.stream_token(f"**💡 解读：**\n{val_data['result_interpretation']}\n\n")
            
            if val_data.get('recommendations'):
                await msg.stream_token(f"**💭 建议：**\n{val_data['recommendations']}\n\n")
        except:
            if validation:
                await msg.stream_token(f"{validation}\n\n")
        
        await msg.stream_token("---\n\n")
        await msg.stream_token("✨ **分析完成！** 您可以继续提出新的问题。\n")
        
        # Reset conversation state
        cl.user_session.set("conversation_state", None)
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        await cl.Message(
            content=f"❌ **执行失败：**\n\n```\n{str(e)}\n```\n\n"
                    f"<details><summary>详细信息</summary>\n\n```\n{error_detail}\n```\n</details>"
        ).send()

async def handle_execution_result(msg, result):
    """处理执行结果（自动执行的情况）"""
    await msg.stream_token("✅ **分析完成！**\n\n")
    
    # Show execution result
    exec_result = result.get('execution_result', '')
    if exec_result:
        await msg.stream_token("**执行结果：**\n")
        await msg.stream_token(f"```\n{exec_result}\n```\n\n")
    
    # Show validation/analysis
    validation = result.get('validation', '')
    if validation:
        try:
            val_data = json.loads(validation)
            
            if val_data.get('final_answer'):
                await msg.stream_token(f"**📌 答案：** {val_data['final_answer']}\n\n")
            
            if val_data.get('result_interpretation'):
                await msg.stream_token(f"**💡 解读：**\n{val_data['result_interpretation']}\n\n")
        except:
            await msg.stream_token(f"{validation}\n\n")

@cl.action_callback("confirm")
async def on_confirm_action(action: cl.Action):
    """处理确认按钮点击"""
    agent = cl.user_session.get("agent")
    profile_data = cl.user_session.get("profile_data")
    conversation_state = cl.user_session.get("conversation_state")
    
    await handle_confirm(agent, profile_data, conversation_state)
    
    # Remove the action buttons
    await action.remove()

@cl.action_callback("modify")
async def on_modify_action(action: cl.Action):
    """处理修改按钮点击"""
    await cl.Message(
        content="💬 请输入您的修改建议..."
    ).send()
    
    await action.remove()

# Add settings
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Data Analysis",
            markdown_description="智能数据分析助手 - 使用 GPT-5-nano, Claude, Gemini 和 GPT-4o-mini",
            icon="📊",
        )
    ]

