# skema_cfm_agents/ui/cli.py

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch the SKEMA CFM Streamlit application."""

    app_path = Path(__file__).resolve().parent / "streamlit_entry.py"

    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            *sys.argv[1:],
        ]
    )
