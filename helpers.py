from __future__ import annotations

from pathlib import Path


def load_env() -> None:
    """Load .env from project root if python-dotenv is available."""
    env_path = Path(__file__).resolve().parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        # Manually parse a simple KEY=value .env file
        if env_path.exists():
            import os
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
