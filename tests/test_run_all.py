"""
Unit tests for master runner scripts/run_all.py arguments and structure.
"""

import subprocess
import sys


def test_run_all_help_flag():
    result = subprocess.run([sys.executable, "scripts/run_all.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--quick" in result.stdout
