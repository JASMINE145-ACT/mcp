import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.cwd() / ".env", override=False)


@dataclass
class Settings:
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1"
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    DEFAULT_AUTHOR: str = "2AIBot"
    DEFAULT_SOURCE_URL: str = ""
    DEFAULT_COVER_PATH: str = ""
    ENABLE_AUTO_PUBLISH: bool = False

    def __post_init__(self) -> None:
        self.OPENAI_API_KEY = self.OPENAI_API_KEY.strip()
        self.OPENAI_MODEL = (self.OPENAI_MODEL or "gpt-4.1").strip()
        self.WECHAT_APP_ID = self.WECHAT_APP_ID.strip()
        self.WECHAT_APP_SECRET = self.WECHAT_APP_SECRET.strip()
        self.DEFAULT_AUTHOR = (self.DEFAULT_AUTHOR or "2AIBot").strip()
        self.DEFAULT_SOURCE_URL = (self.DEFAULT_SOURCE_URL or "").strip()
        self.DEFAULT_COVER_PATH = (self.DEFAULT_COVER_PATH or "").strip()
        if isinstance(self.ENABLE_AUTO_PUBLISH, str):
            self.ENABLE_AUTO_PUBLISH = self.ENABLE_AUTO_PUBLISH.strip().lower() == "true"

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未配置，请在 .env 文件中填写。")
        if not self.WECHAT_APP_ID:
            raise ValueError("WECHAT_APP_ID 未配置，请在 .env 文件中填写。")
        if not self.WECHAT_APP_SECRET:
            raise ValueError("WECHAT_APP_SECRET 未配置，请在 .env 文件中填写。")


def get_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", ""),
        OPENAI_MODEL=os.environ.get("OPENAI_MODEL", "gpt-4.1"),
        WECHAT_APP_ID=os.environ.get("WECHAT_APP_ID", ""),
        WECHAT_APP_SECRET=os.environ.get("WECHAT_APP_SECRET", ""),
        DEFAULT_AUTHOR=os.environ.get("DEFAULT_AUTHOR", "2AIBot"),
        DEFAULT_SOURCE_URL=os.environ.get("DEFAULT_SOURCE_URL", ""),
        DEFAULT_COVER_PATH=os.environ.get("DEFAULT_COVER_PATH", ""),
        ENABLE_AUTO_PUBLISH=get_enable_auto_publish(),
    )


def load_style_guide(style_path: str) -> str:
    path = Path(style_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"写作风格文件不存在: {path}")
    return path.read_text(encoding="utf-8")


def get_wechat_app_id() -> str:
    v = os.environ.get("WECHAT_APP_ID", "").strip()
    if not v:
        raise SystemExit("WECHAT_APP_ID 未配置，请在 .env 文件中填写。")
    return v


def get_wechat_app_secret() -> str:
    v = os.environ.get("WECHAT_APP_SECRET", "").strip()
    if not v:
        raise SystemExit("WECHAT_APP_SECRET 未配置，请在 .env 文件中填写。")
    return v


def get_default_author() -> str:
    return os.environ.get("DEFAULT_AUTHOR", "2AIBot")


def get_default_cover_path() -> str:
    return os.environ.get("DEFAULT_COVER_PATH", "")


def get_default_source_url() -> str:
    return os.environ.get("DEFAULT_SOURCE_URL", "")


def get_enable_auto_publish() -> bool:
    return os.environ.get("ENABLE_AUTO_PUBLISH", "false").strip().lower() == "true"
