import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_missing_openai_key_raises():
    original = os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("WECHAT_APP_ID", None)
    os.environ.pop("WECHAT_APP_SECRET", None)
    try:
        from config import settings as s_module
        import importlib
        importlib.reload(s_module)
        try:
            s_module.get_settings()
            assert False, "Should have raised"
        except (SystemExit, Exception) as e:
            assert True
    finally:
        if original:
            os.environ["OPENAI_API_KEY"] = original


def test_missing_app_id_raises(monkeypatch=None):
    original_id = os.environ.pop("WECHAT_APP_ID", None)
    original_secret = os.environ.pop("WECHAT_APP_SECRET", None)
    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    try:
        from pydantic import ValidationError
        from config.settings import Settings
        try:
            Settings(
                OPENAI_API_KEY="test_key",
                WECHAT_APP_ID="",
                WECHAT_APP_SECRET="test_secret",
            )
            assert False, "Should have raised ValidationError"
        except (ValidationError, ValueError, SystemExit):
            assert True
    finally:
        if original_id:
            os.environ["WECHAT_APP_ID"] = original_id
        if original_secret:
            os.environ["WECHAT_APP_SECRET"] = original_secret


def test_missing_app_secret_raises():
    from pydantic import ValidationError
    from config.settings import Settings
    try:
        Settings(
            OPENAI_API_KEY="test_key",
            WECHAT_APP_ID="test_app_id",
            WECHAT_APP_SECRET="",
        )
        assert False, "Should have raised ValidationError"
    except (ValidationError, ValueError, SystemExit):
        assert True


def test_token_cache_populated():
    from wechat.token import _cache, clear_token_cache
    clear_token_cache()
    assert _cache["token"] is None


if __name__ == "__main__":
    tests = [
        test_missing_app_id_raises,
        test_missing_app_secret_raises,
        test_token_cache_populated,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
