import sys
from pathlib import Path

# Make project root importable so `import serving.main` works from any test
sys.path.insert(0, str(Path(__file__).parent.parent))
