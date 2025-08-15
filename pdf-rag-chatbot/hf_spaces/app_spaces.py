import runpy
import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Execute the main Streamlit app
# Equivalent to: streamlit run app.py (but within same Python process)
runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
