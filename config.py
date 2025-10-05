# config.py
import os

def get_secret(name: str, section: str = None):
    """Get secret from Streamlit secrets or environment variable"""
    try:
        import streamlit as st
        if section:
            return (st.secrets.get(section) or {}).get(name) or os.getenv(name)
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)

def require_secret(name: str, section: str = None):
    """Get required secret, stop if missing"""
    v = get_secret(name, section)
    if not v:
        try:
            import streamlit as st
            st.error(f"Missing secret: {name}")
            st.stop()
        except:
            # If not in Streamlit context, raise error
            raise ValueError(f"Missing required secret: {name}")
    return v
