"""Prediction page for this project.  python app.py  (or: mlguy serve)"""

from pathlib import Path

from mlguy.app import launch

if __name__ == "__main__":
    launch(Path(__file__).parent)
