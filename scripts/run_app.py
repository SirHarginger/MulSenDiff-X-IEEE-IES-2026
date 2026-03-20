#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the MulSenDiff-X Streamlit app.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "app" / "streamlit_app.py"),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])
    if args.streamlit_args:
        command.extend(args.streamlit_args)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()

