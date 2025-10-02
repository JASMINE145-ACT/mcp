# 🔍 LangSmith 监控集成指南

## 什么是 LangSmith？

LangSmith 是 LangChain 提供的强大监控和调试工具，可以帮你：

- 📊 **可视化追踪**：查看每次对话的完整 LLM 调用链
- 💰 **成本监控**：追踪每个调用的 token 使用和费用
- ⏱️ **性能分析**：监控响应时间、延迟、错误率
- 🐛 **调试工具**：查看每一步的输入输出
- 📈 **数据分析**：长期使用趋势和性能优化建议

## 🚀 快速开始

### 1. 注册 LangSmith 账号

1. 访问：https://smith.langchain.com/
2. 使用 GitHub 或 Google 账号注册（免费）
3. 创建一个新项目（例如：`data-analysis-agent`）

### 2. 获取 API Key

1. 登录后，点击右上角头像
2. 进入 **Settings** → **API Keys**
3. 点击 **Create API Key**
4. 复制生成的 API Key

### 3. 配置环境变量

在你的 `.env` 文件中添加：

```bash
# 已有的配置
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# ✅ 新增：LangSmith 配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls-xxxxx-your-actual-key
LANGCHAIN_PROJECT=data-analysis-agent

# 可选：启用详细调试
# LANGCHAIN_DEBUG=true
```

### 4. 安装依赖

```bash
pip install langsmith>=0.1.0
```

### 5. 运行并查看追踪

启动你的应用：

```bash
# CLI 模式
python agent.py

# Streamlit 模式
streamlit run streamlit_app.py
```

初始化时会看到：
```
✅ LangSmith tracing enabled - Project: data-analysis-agent
   View at: https://smith.langchain.com/
```

然后访问 https://smith.langchain.com/ 查看实时追踪！

## 📊 在 LangSmith Dashboard 中能看到什么

### 1. 调用链追踪

每次用户提问，你会看到完整的调用流程：

```
用户问题: "画一个直方图"
├─ GPT-5-nano: 生成分析计划
│  ├─ 输入: 问题 + 数据信息
│  ├─ 输出: JSON 计划
│  ├─ Token: 1,234 tokens
│  └─ 延迟: 1.2s
├─ Claude: 生成 Python 代码
│  ├─ 输入: 计划 + 数据信息
│  ├─ 输出: Python 代码
│  ├─ Token: 856 tokens
│  └─ 延迟: 2.1s
├─ 本地执行: 运行代码
│  └─ 输出: 图表文件
└─ GPT-4o-mini: 分析验证
   ├─ 输入: 问题 + 执行结果
   ├─ 输出: 分析解读
   ├─ Token: 678 tokens
   └─ 延迟: 0.8s

总计: 2,768 tokens | 4.1s | $0.012
```

### 2. 性能指标

- **响应时间分布图**：P50, P95, P99 延迟
- **错误率**：成功/失败比例
- **Token 使用趋势**：每日/每周消耗
- **成本追踪**：实时费用统计

### 3. 调试功能

- **查看原始请求/响应**：完整的 prompt 和输出
- **对比不同版本**：A/B 测试效果
- **标记问题案例**：收集改进数据
- **搜索和过滤**：按时间、用户、错误筛选

## 💡 最佳实践

### 1. 使用有意义的项目名

```bash
LANGCHAIN_PROJECT=data-analysis-prod     # 生产环境
LANGCHAIN_PROJECT=data-analysis-dev      # 开发环境
LANGCHAIN_PROJECT=data-analysis-test     # 测试环境
```

### 2. 添加自定义标签

在代码中添加元数据：

```python
from langsmith import traceable

@traceable(
    name="generate_plot",
    tags=["visualization", "production"],
    metadata={"user_id": "user123"}
)
def generate_plot_code(plan):
    # your code here
    pass
```

### 3. 设置成本预算

在 LangSmith 中设置每日/每月预算警报，避免意外高额费用。

### 4. 定期审查

- 每周查看慢查询
- 识别高成本调用
- 优化 prompt 以减少 token

## 🔒 隐私和安全

### 数据隐私

LangSmith 会记录：
- ✅ LLM 调用的输入输出
- ✅ Token 使用和延迟
- ⚠️ 可能包含敏感数据（如用户问题、数据内容）

### 如果处理敏感数据：

1. **使用私有部署**：LangSmith 支持自托管
2. **关闭追踪**：生产环境设置 `LANGCHAIN_TRACING_V2=false`
3. **过滤敏感信息**：在发送前清理数据

## 🎯 实用场景

### 场景 1: 调试慢响应

问题：某些问题响应很慢

解决：
1. 在 LangSmith 中按延迟排序
2. 发现 Claude 代码生成步骤耗时最长
3. 优化 prompt，减少不必要的说明
4. 响应时间从 5s 降到 2s

### 场景 2: 减少成本

问题：每月 API 成本过高

解决：
1. 查看 token 使用统计
2. 发现 GPT-5-nano 每次消耗 2000+ tokens
3. 精简 dataset_info，只传必要字段
4. 成本降低 40%

### 场景 3: 提高准确率

问题：有些问题分析不准确

解决：
1. 在 LangSmith 中标记失败案例
2. 导出这些案例的 prompt 和输出
3. 优化 prompt 模板
4. 准确率从 85% 提升到 95%

## 📚 高级功能

### 1. Datasets & Testing

创建测试数据集，自动评估模型性能：

```python
from langsmith import Client

client = Client()

# 创建测试数据集
examples = [
    {"input": "画一个直方图", "expected": "histogram"},
    {"input": "计算平均值", "expected": "calculation"},
]

dataset = client.create_dataset("test-questions")
for ex in examples:
    client.create_example(**ex, dataset_id=dataset.id)
```

### 2. 自动评估

设置自动评估规则，每次运行后自动打分。

### 3. Webhooks

设置 webhook，在特定事件时触发通知：
- 错误率超过阈值
- 响应时间过长
- 成本超预算

## 🆘 常见问题

### Q: LangSmith 是免费的吗？

A: 
- **个人版**：免费，每月 5,000 次追踪
- **专业版**：$39/月，无限追踪
- **企业版**：联系销售

### Q: 会增加响应延迟吗？

A: 几乎不会。追踪是异步的，通常只增加 <10ms。

### Q: 如何临时关闭追踪？

A: 设置 `LANGCHAIN_TRACING_V2=false` 或删除该环境变量。

### Q: 可以追踪非 LangChain 的 LLM 调用吗？

A: 可以，使用 `@traceable` 装饰器包装任何函数。

## 🔗 相关资源

- **官方文档**：https://docs.smith.langchain.com/
- **视频教程**：https://www.youtube.com/@LangChain
- **Discord 社区**：https://discord.gg/langchain
- **GitHub**：https://github.com/langchain-ai/langsmith-sdk

## 🎉 总结

LangSmith 是监控和优化 LLM 应用的必备工具。只需：

1. ✅ 注册账号
2. ✅ 添加 4 行环境变量
3. ✅ 无需改代码

就能获得：

- 📊 完整的可观测性
- 💰 成本控制
- 🐛 强大的调试能力
- 📈 持续优化的数据支持

立即开始：https://smith.langchain.com/ 🚀

