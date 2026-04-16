"""
FrontEnd/utils_frontend.py
Lightweight UI-side helpers (no pandas/numpy heavy imports).
Mirrors the helpers in BackEnd/utils.py for display purposes only.
"""
import re
import numpy as np
import pandas as pd


def to_client_steps(text_or_list) -> list:
    steps = []
    if isinstance(text_or_list, list):
        steps = [s for s in text_or_list if s and isinstance(s, str)]
    elif isinstance(text_or_list, str):
        raw = re.split(r'\s*[\.\n;]\s*', text_or_list.strip())
        steps = [s for s in raw if s]
    steps = steps[:4]
    # Strip any existing "Step N:" prefix the LLM may have already added
    cleaned = [re.sub(r"^Step\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE).rstrip(".") for s in steps]
    return [f"Step {i + 1}: {s}" for i, s in enumerate(cleaned)]


def format_calc_value(val) -> str:
    if val is None or val == "":
        return "needs_data"
    if isinstance(val, (int, float)):
        return f"{val}"
    return str(val)
