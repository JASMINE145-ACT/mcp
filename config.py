# config.py
import os
def get_secret(name: str, section: str | None = None):
    try:
        import streamlit as st
        if section:
            return (st.secrets.get(section) or {}).get(name) or os.getenv(name)
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)

def require_secret(name, section=None):
    v = get_secret(name, section)
    if not v:
        import streamlit as st
        st.error(f"Missing secret: {name}")
        st.stop()
    return v

