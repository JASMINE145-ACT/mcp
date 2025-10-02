import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import glob
from PIL import Image
from agent import DataAIAgent
from function import profile_dataframe_simple

# 页面配置
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = None
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'waiting_for_confirm' not in st.session_state:
    st.session_state.waiting_for_confirm = False
if 'pending_plan' not in st.session_state:
    st.session_state.pending_plan = None

# 侧边栏 - 页面导航
st.sidebar.title("📊 AI Data Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "选择页面",
    ["📁 上传数据", "💬 AI 对话", "📈 查看结果"],
    index=0 if st.session_state.df is None else 1
)

st.sidebar.markdown("---")

# 显示当前数据集信息
if st.session_state.df is not None:
    st.sidebar.success("✅ 数据已加载")
    st.sidebar.metric("行数", st.session_state.df.shape[0])
    st.sidebar.metric("列数", st.session_state.df.shape[1])
else:
    st.sidebar.warning("⚠️ 请先上传数据")

# ============================================================================
# 页面 1: 上传数据
# ============================================================================
if page == "📁 上传数据":
    st.title("📁 数据上传")
    st.markdown("上传 CSV 文件开始分析")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "选择 CSV 文件",
            type=['csv'],
            help="支持 CSV 格式文件"
        )
        
        if uploaded_file is not None:
            try:
                # 读取数据
                df = pd.read_csv(uploaded_file)
                
                # 显示数据预览
                st.success(f"✅ 成功加载文件：{uploaded_file.name}")
                st.info(f"📊 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
                
                # 数据预览
                st.subheader("数据预览")
                st.dataframe(df.head(20), use_container_width=True)
                
                # 基本统计信息
                st.subheader("基本统计")
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.write("**数值列统计**")
                    st.dataframe(df.describe(), use_container_width=True)
                
                with col_stat2:
                    st.write("**列信息**")
                    info_df = pd.DataFrame({
                        '列名': df.columns,
                        '类型': df.dtypes.astype(str),
                        '非空数': df.count().values,
                        '空值数': df.isnull().sum().values
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                # 确认按钮
                if st.button("🚀 使用此数据集", type="primary", use_container_width=True):
                    # 保存到 session state
                    st.session_state.df = df
                    
                    # 生成数据 profile
                    with st.spinner("正在分析数据结构..."):
                        st.session_state.profile_data = profile_dataframe_simple(df)
                    
                    # 初始化 agent
                    with st.spinner("正在初始化 AI Agent..."):
                        st.session_state.agent = DataAIAgent(df)
                    
                    # 重置对话状态
                    st.session_state.conversation_state = None
                    st.session_state.analysis_results = []
                    st.session_state.waiting_for_confirm = False
                    st.session_state.pending_plan = None
                    
                    st.success("✅ 数据集加载成功！现在可以开始对话了")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ 加载文件失败：{str(e)}")
    
    with col2:
        st.info("""
        ### 📝 使用说明
        
        1. 上传 CSV 文件
        2. 查看数据预览
        3. 确认使用数据集
        4. 前往 AI 对话页面
        
        ### 💡 提示
        - 支持中文列名
        - 建议文件大小 < 100MB
        - 确保数据格式正确
        """)

# ============================================================================
# 页面 2: AI 对话
# ============================================================================
elif page == "💬 AI 对话":
    st.title("💬 AI 数据分析对话")
    
    if st.session_state.df is None or st.session_state.agent is None:
        st.warning("⚠️ 请先上传数据集")
        if st.button("前往上传页面"):
            st.rerun()
    else:
        # 显示可用列信息
        with st.expander("📊 数据集信息", expanded=False):
            col_list = st.session_state.df.columns.tolist()
            st.write(f"**可用列（{len(col_list)}）：**")
            st.write(", ".join([f"`{col}`" for col in col_list[:20]]))
            if len(col_list) > 20:
                st.write(f"... 还有 {len(col_list) - 20} 列")
        
        st.markdown("---")
        
        # 对话界面
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            # 如果正在等待确认
            if st.session_state.waiting_for_confirm and st.session_state.pending_plan:
                st.info("📋 AI 已经为你制定了分析计划，请确认后执行")
                
                # 显示计划
                plan_data = st.session_state.pending_plan
                
                with st.container():
                    st.markdown("### 📋 分析计划")
                    
                    try:
                        plan_json = json.loads(plan_data.get('plan', '{}'))
                        analysis_plan = plan_json.get('analysis_plan', plan_json)
                        
                        st.write(f"**💬 计划说明：** {plan_data.get('message_to_user', 'N/A')}")
                        st.write(f"**🎯 分析类型：** {analysis_plan.get('analysis_type', 'N/A')}")
                        st.write(f"**📊 使用列：** {', '.join(analysis_plan.get('columns_needed', []))}")
                        st.write(f"**🔧 方法：** {analysis_plan.get('method', 'N/A')}")
                        
                        if plan_data.get('optional_suggestions'):
                            st.info(f"💡 **建议：** {plan_data.get('optional_suggestions')}")
                    except:
                        st.write(plan_data.get('plan', 'N/A'))
                
                # 确认按钮
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                
                with col_btn1:
                    if st.button("✅ 确认执行", type="primary", use_container_width=True):
                        with st.spinner("🔄 执行中..."):
                            # 确认并执行
                            st.session_state.conversation_state['plan_confirmed'] = True
                            st.session_state.conversation_state['user_feedback'] = "用户确认计划，请执行"
                            
                            # 执行分析
                            result = st.session_state.agent.langgraph_analysis(
                                question=st.session_state.current_question,
                                dataset_info=st.session_state.profile_data,
                                user_feedback=st.session_state.conversation_state['user_feedback'],
                                existing_state=st.session_state.conversation_state
                            )
                            
                            # 保存结果
                            st.session_state.analysis_results.append({
                                'question': st.session_state.current_question,
                                'result': result
                            })
                            
                            # 重置状态
                            st.session_state.waiting_for_confirm = False
                            st.session_state.pending_plan = None
                            st.session_state.conversation_state = None
                            
                            st.success("✅ 分析完成！")
                            st.rerun()
                
                with col_btn2:
                    if st.button("❌ 重新规划", use_container_width=True):
                        st.session_state.waiting_for_confirm = False
                        st.session_state.pending_plan = None
                        st.session_state.conversation_state = None
                        st.rerun()
                
                # 反馈输入
                feedback = st.text_input("💬 或者提供反馈来优化计划", placeholder="例如：换一个列，使用不同的图表类型...")
                if feedback:
                    if st.button("📤 发送反馈"):
                        with st.spinner("🔄 处理反馈中..."):
                            result = st.session_state.agent.langgraph_analysis(
                                question=st.session_state.current_question,
                                dataset_info=st.session_state.profile_data,
                                user_feedback=feedback,
                                existing_state=st.session_state.conversation_state
                            )
                            
                            st.session_state.conversation_state = result.get('conversation_state', {})
                            st.session_state.pending_plan = result
                            st.rerun()
            
            else:
                # 问题输入
                question = st.text_area(
                    "🙋 请输入你的问题",
                    height=100,
                    placeholder="例如：\n- 画一个价格的直方图\n- 计算每个类别的平均值\n- 显示销量前10名",
                    key="question_input"
                )
                
                col_submit, col_clear = st.columns([1, 3])
                
                with col_submit:
                    if st.button("🚀 开始分析", type="primary", use_container_width=True):
                        if question.strip():
                            st.session_state.current_question = question.strip()
                            
                            with st.spinner("🤔 AI 正在思考..."):
                                # 调用 agent 分析
                                result = st.session_state.agent.langgraph_analysis(
                                    st.session_state.current_question,
                                    st.session_state.profile_data
                                )
                                
                                conv_state = result.get('conversation_state', {})
                                
                                # 检查是否需要确认
                                if conv_state.get('needs_user_input', False):
                                    st.session_state.waiting_for_confirm = True
                                    st.session_state.pending_plan = result
                                    st.session_state.conversation_state = {
                                        'question': st.session_state.current_question,
                                        'dataset_info': st.session_state.profile_data,
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
                                    st.rerun()
                        else:
                            st.warning("请输入问题")
                
                with col_clear:
                    if st.button("🗑️ 清空历史", use_container_width=True):
                        st.session_state.analysis_results = []
                        st.session_state.conversation_state = None
                        st.session_state.waiting_for_confirm = False
                        st.session_state.pending_plan = None
                        st.rerun()
        
        with col_side:
            st.markdown("### 💡 快速开始")
            
            example_questions = [
                "画一个直方图",
                "计算平均值",
                "显示前10行数据",
                "统计每个类别的数量",
                "创建散点图",
                "计算相关系数"
            ]
            
            st.write("**示例问题（点击使用）：**")
            for eq in example_questions:
                if st.button(f"💬 {eq}", key=f"example_{eq}", use_container_width=True):
                    st.session_state.current_question = eq
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 分析历史")
            st.metric("已完成分析", len(st.session_state.analysis_results))

# ============================================================================
# 页面 3: 查看结果
# ============================================================================
elif page == "📈 查看结果":
    st.title("📈 分析结果")
    
    if not st.session_state.analysis_results:
        st.info("💡 还没有分析结果，请先在 AI 对话页面提问")
        if st.button("前往 AI 对话"):
            st.rerun()
    else:
        # 结果列表
        st.sidebar.markdown("### 📋 结果列表")
        
        result_titles = [
            f"{i+1}. {r['question'][:30]}..." 
            for i, r in enumerate(st.session_state.analysis_results)
        ]
        
        selected_idx = st.sidebar.radio(
            "选择查看的结果",
            range(len(st.session_state.analysis_results)),
            format_func=lambda x: result_titles[x]
        )
        
        # 清空按钮
        if st.sidebar.button("🗑️ 清空所有结果", use_container_width=True):
            st.session_state.analysis_results = []
            st.rerun()
        
        # 显示选中的结果
        selected_result = st.session_state.analysis_results[selected_idx]
        
        st.markdown(f"## 问题 {selected_idx + 1}")
        st.info(f"**🙋 问题：** {selected_result['question']}")
        
        result = selected_result['result']
        
        # Tab 布局
        tab1, tab2, tab3, tab4 = st.tabs(["📊 结果", "💻 代码", "📋 计划", "🔍 详情"])
        
        with tab1:
            
            # 显示生成的图片
            st.markdown("### 📈 生成的图表")
            png_files = glob.glob("*.png")
            if png_files:
                # 按修改时间排序，显示最新的
                png_files.sort(key=os.path.getmtime, reverse=True)
                
                cols = st.columns(2)
                for idx, png_file in enumerate(png_files[:4]):  # 最多显示4张
                    with cols[idx % 2]:
                        try:
                            image = Image.open(png_file)
                            st.image(image, caption=png_file, use_column_width=True)
                        except:
                            st.warning(f"无法加载图片：{png_file}")
            else:
                st.info("没有生成图表")
            
            st.markdown("---")
            st.markdown("### 🔍 AI 分析与解读")
            
            validation = result.get('validation', 'N/A')
            
            # 直接显示自然语言文本，不解析 JSON
            if validation and validation != 'N/A':
                st.markdown(validation)
            else:
                st.info("暂无分析结果")
        
        with tab2:
            st.markdown("### 💻 生成的代码")
            code = result.get('code', 'N/A')
            st.code(code, language='python')
            
            # 下载按钮
            st.download_button(
                "📥 下载代码",
                code,
                file_name=f"analysis_{selected_idx+1}.py",
                mime="text/plain"
            )
        
        with tab3:
            st.markdown("### 📋 分析计划")
            plan = result.get('plan', 'N/A')
            
            try:
                plan_json = json.loads(plan)
                st.json(plan_json)
            except:
                st.code(plan, language='text')
        
        with tab4:
            st.markdown("### 🔍 完整结果")
            st.json(result)

# ============================================================================
# 页脚
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>AI Data Analysis Agent v1.0</p>
    <p>Powered by Claude & GPT</p>
</div>
""", unsafe_allow_html=True)

