#!/usr/bin/env python3
"""Runs every Z3 proof script in this directory and exits nonzero if any
expected-PROVED property fails.

Notes:
  - euler_jacobian.py is run in its default (reference-formula) mode here,
    which is expected to be ALL PROVED. Run
    `python3 euler_jacobian.py --source` separately to see the known
    P_roll_Rotate[1][2] bug reproduced as a COUNTEREXAMPLE against the
    verbatim source transcription (see README.md).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "quat_jacobian.py",
    "euler_jacobian.py",          # reference (correct-formula) mode
    "eskf_attitude_jacobian.py",
    "pixel_mapping.py",
    "slam_index_bookkeeping.py",
]


def main():
    overall_ok = True
    for script in SCRIPTS:
        path = HERE / script
        print(f"\n===== {script} =====")
        result = subprocess.run([sys.executable, str(path)], cwd=str(HERE))
        ok = (result.returncode == 0)
        overall_ok = overall_ok and ok
        print(f"----- {script}: {'OK' if ok else 'FAILED'} (exit={result.returncode}) -----")

    print("\n=====================================")
    if overall_ok:
        print("ALL SCRIPTS PROVED THEIR EXPECTED PROPERTIES")
    else:
        print("AT LEAST ONE SCRIPT FAILED -- see output above")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
