import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL {message}")


def ok(message: str) -> None:
    print(f"OK   {message}")


def main() -> int:
    failures: list[str] = []

    required_files = [
        "README.md",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
        "server.py",
        "mcp_entry.py",
        "run_server.bat",
        "start.ps1",
        "docs/quickstart.md",
        "docs/mcp_client_config.md",
        "docs/release_checklist.md",
        "examples/mcp-config.source.json",
        "examples/mcp-config.installed.json",
    ]
    for rel in required_files:
        if (ROOT / rel).exists():
            ok(f"{rel} exists")
        else:
            fail(f"{rel} is missing", failures)

    env_example = ROOT / ".env.example"
    if env_example.exists():
        text = env_example.read_text(encoding="utf-8")
        for key in ["WECHAT_APP_ID", "WECHAT_APP_SECRET", "OPENAI_API_KEY"]:
            marker = f"{key}="
            lines = [line for line in text.splitlines() if line.startswith(marker)]
            if not lines:
                fail(f".env.example missing {key}", failures)
            elif lines[0].strip() != marker and key != "OPENAI_MODEL":
                fail(f".env.example should not contain a real value for {key}", failures)
            else:
                ok(f".env.example has placeholder for {key}")

    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        ignored = gitignore.read_text(encoding="utf-8")
        for pattern in [".env", "storage/drafts/", "storage/images/", "*.log"]:
            if pattern in ignored:
                ok(f".gitignore protects {pattern}")
            else:
                fail(f".gitignore should protect {pattern}", failures)

    if (ROOT / ".env").exists():
        print("WARN .env exists locally. Do not include it when sharing the project.")

    if failures:
        print(f"\n{len(failures)} release checks failed.")
        return 1
    print("\nRelease checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
