import sys
from pathlib import Path

# Modules import siblings as experiments.nsd_replication.*; tests run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
