"""
多公众号账号 profile 管理。

Profile YAML 格式：
  account_name: 印中商务中心
  default_author: CBC
  default_template: A
  tone: 商务、专业、克制
  target_reader: 中国出海企业老板/高管
  preferred_structure:
    - 事件背景
    - 核心信息
    - 对中企影响
    - 建议
  banned_words:
    - 震惊
    - 爆了
"""
import os
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

BASE_DIR = Path(__file__).parent.parent
PROFILES_DIR = BASE_DIR / "config" / "profiles"

# Built-in defaults when no profile file exists
_DEFAULTS: dict[str, dict] = {
    "default": {
        "account_name": "默认账号",
        "default_author": os.environ.get("DEFAULT_AUTHOR", "2AIBot"),
        "default_template": "A",
        "tone": "专业、客观",
        "target_reader": "通用读者",
        "preferred_structure": ["背景", "核心内容", "总结"],
        "banned_words": ["震惊", "爆了", "暴富", "不敢相信"],
    }
}


def list_profiles() -> list[str]:
    """Return available profile names (from YAML files + built-in defaults)."""
    names = set(_DEFAULTS.keys())
    if PROFILES_DIR.exists():
        for f in PROFILES_DIR.glob("*.yaml"):
            names.add(f.stem)
        for f in PROFILES_DIR.glob("*.yml"):
            names.add(f.stem)
    return sorted(names)


def load_profile(name: str = "default") -> dict:
    """Load a profile by name. Falls back to built-in default if not found."""
    if PROFILES_DIR.exists() and _HAS_YAML:
        for ext in (".yaml", ".yml"):
            p = PROFILES_DIR / f"{name}{ext}"
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # Fill in any missing fields from defaults
                base = dict(_DEFAULTS.get("default", {}))
                base.update(data)
                return base

    if name in _DEFAULTS:
        return dict(_DEFAULTS[name])

    return dict(_DEFAULTS["default"])


def get_current_profile() -> dict:
    """Return the active profile based on WECHAT_PROFILE env var, or default."""
    profile_name = os.environ.get("WECHAT_PROFILE", "default").strip()
    return load_profile(profile_name)


def create_profile_template(name: str) -> Path:
    """Create a YAML template file for a new profile. Returns the file path."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    p = PROFILES_DIR / f"{name}.yaml"
    if p.exists():
        return p
    template = f"""# Profile: {name}
account_name: {name}
default_author: 2AIBot
default_template: A   # A=蓝色商业分析 B=蓝色财经科普 C=紫色新闻资讯 D=钢蓝深度评论
tone: 专业、客观
target_reader: 目标读者描述

preferred_structure:
  - 背景
  - 核心内容
  - 影响与建议
  - 总结

banned_words:
  - 震惊
  - 爆了
  - 不敢相信
"""
    p.write_text(template, encoding="utf-8")
    return p
