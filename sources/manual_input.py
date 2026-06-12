def get_topic_from_prompt() -> str:
    """Interactive prompt for getting a topic from stdin."""
    topic = input("请输入文章主题: ").strip()
    if not topic:
        raise ValueError("主题不能为空")
    return topic
