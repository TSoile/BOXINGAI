# BoKing 🥊

BoKing is a simple GUI application that analyzes boxing sparring footage.

## What it does

- Lets a user upload a sparring video.
- Tracks two fighters across the footage.
- Estimates punch attempts and punches landed using motion-based heuristics.
- Returns summary stats and timeline data (JSON + CSV download).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- This is an MVP for analysis, not an official judging/scoring system.
- Punch detection is heuristic and should be calibrated for your camera angle and video quality.
- Use the **Close BoKing App** button in the sidebar (or `Ctrl + C` in terminal) to stop the app.
