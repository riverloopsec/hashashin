import subprocess
from pathlib import Path
import sys

def _run(bash_script, *args):
    subprocess.run(["bash", str(bash_script)] + [str(_) for _ in args])

def safedocs_demo():
    _run(Path(__file__).parent / "safedocs_05_16_demo.sh", sys.argv[1:])