import sys
import os

# Add the current directory to sys.path so we can import pipeline
sys.path.append(os.getcwd())

try:
    from pipeline.pipeline import run_pipeline
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")