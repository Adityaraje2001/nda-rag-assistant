# app.py (root)
import os
import sys

# Load .env locally; on Spaces, set Secrets in the Space settings
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure we can import the src package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Optional sanity log to confirm envs are visible at runtime
print(
    "APP DEBUG:",
    "model:", os.getenv("ROUTER_MODEL"),
    "base:", os.getenv("ROUTER_BASE_URL"),
    "key?", bool(os.getenv("ROUTER_API_KEY")),
)

# Import the Gradio Blocks instance defined at module scope in src/app_rag.py
from src.app_rag import demo  # src/app_rag.py must define `demo` at module scope

if __name__ == "__main__":
    # For a public URL during local testing, use: demo.launch(share=True)
    demo.launch()
