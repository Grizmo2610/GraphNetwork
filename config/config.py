import os
import numpy as np


INK = "#0B0E11"; PANEL = "#12161C"; LINE = "#232B34"; TEXT = "#D7DEE6"; MUTED = "#7C8A99"
GOOD = "#33D6A6"; BAD = "#F0654A"; ACCENT = "#4FA8FF"; ACCENT2 = "#C792EA"

SEED = 42

RNG: np.random.Generator = np.random.default_rng(SEED)

# PATH
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
all_metric_path = os.path.join(OUTPUT_FOLDER, "metrics_all.json")

# 
SCALE_CONFIG = {
    "SMALL":  dict(n_customers=800),
    "MEDIUM": dict(n_customers=3200),
    "LARGE":  dict(n_customers=12800),
}

