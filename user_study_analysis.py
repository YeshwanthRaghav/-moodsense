"""
MoodSense User Study Analysis
==============================
Run AFTER collecting user study data via Google Forms.

Input:  user_study_data.csv  (exported from Google Forms)
        moodsense_results.json (from moodsense_evaluation.py)

Expected CSV columns (rename your Form columns to match):
  participant_id, age, gender, phq2_score, journal_text_1..5,
  helpfulness_rating (1-5), mood_before (1-10), mood_after (1-10)

Output: Prints Section IV-D results ready to paste into the IEEE paper.

Ethics reminder:
  - Obtain IRB/ethics approval BEFORE collecting data
  - Store participant data securely; do not commit CSV to GitHub
  - Add your IRB protocol number to Section IV-D of the paper
"""

import json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CSV_PATH     = "user_study_data.csv"   # <-- put your Google Forms export here
RESULTS_JSON = "moodsense_results.json"
WEIGHTS      = (0.5, 0.3, 0.2)        # MoodSense default fusion weights

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
import torch
device = 0 if torch.cuda.is_available() else -1
bert_pipe = pipeline("text-classification",
                     model="distilbert-base-uncased-finetuned-sst-2-english",
                     device=device)
vader = SentimentIntensityAnalyzer()

def ensemble_score(text, w=WEIGHTS):
    r = bert_pipe(text[:512], truncation=True)[0]
    b = r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]
    v = (vader.polarity_scores(text)["compound"] + 1) / 2
    t = (TextBlob(text).sentiment.polarity + 1) / 2
    return (w[0]*b + w[1]*v + w[2]*t) * 100

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"❌ {CSV_PATH} not found.")
    print("   Export your Google Forms responses as CSV and place it here.")
    sys.exit(1)

required_cols = ["participant_id","age","gender","phq2_score",
                 "journal_text_1","helpfulness_rating","mood_before","mood_after"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print("   Rename your Google Forms columns to match the expected names above.")
    sys.exit(1)

print(f"✅ Loaded {len(df)} participants from {CSV_PATH}")

# ─────────────────────────────────────────────
# Compute MoodSense WB scores per participant
# ─────────────────────────────────────────────
journal_cols = [c for c in df.columns if c.startswith("journal_text_")]
print(f"Found {len(journal_cols)} journal entry columns: {journal_cols}")

wb_scores = []
for _, row in df.iterrows():
    entries = [str(row[c]) for c in journal_cols if pd.notna(row[c]) and str(row[c]).strip()]
    if entries:
        wb_scores.append(np.mean([ensemble_score(e) for e in entries]))
    else:
        wb_scores.append(np.nan)

df["wb_score"] = wb_scores

# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────
valid = df.dropna(subset=["wb_score","phq2_score","helpfulness_rating","mood_before","mood_after"])
n = len(valid)

# PHQ-2 correlation (note: higher PHQ-2 = more depressed → inverse of WB)
spear_r, spear_p = spearmanr(valid["wb_score"], -valid["phq2_score"])

# Helpfulness
helpful_pct = (valid["helpfulness_rating"] >= 4).mean() * 100

# Mood improvement
mood_diff = valid["mood_after"] - valid["mood_before"]
mood_mean = mood_diff.mean()
_, wilcox_p = wilcoxon(mood_diff)

# Demographics
mean_age = valid["age"].mean()
sd_age   = valid["age"].std()
female_pct = (valid["gender"].str.lower().str.strip() == "female").mean() * 100

# ─────────────────────────────────────────────
# Print Section IV-D results
# ─────────────────────────────────────────────
SEP = "=" * 65
print(f"\n{SEP}")
print("SECTION IV-D — PASTE INTO PAPER")
print(SEP)
print(f"n                   = {n}")
print(f"Mean age            = {mean_age:.1f} (SD={sd_age:.1f})")
print(f"% female            = {female_pct:.0f}%")
print(f"Spearman r (WB~PHQ) = {spear_r:.3f}  (p={spear_p:.4f})")
print(f"Helpful ≥4/5        = {helpful_pct:.0f}%")
print(f"Mood improvement    = {mood_mean:+.1f} pts  (Wilcoxon p={wilcox_p:.4f})")
print(SEP)
print("\n✅ Copy the values above into Section IV-D of your IEEE paper.")

# ─────────────────────────────────────────────
# Append to moodsense_results.json
# ─────────────────────────────────────────────
try:
    with open(RESULTS_JSON) as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

results["user_study"] = {
    "n":                n,
    "mean_age":         round(mean_age, 1),
    "sd_age":           round(sd_age,   1),
    "female_pct":       round(female_pct, 0),
    "spearman_r":       round(spear_r,  3),
    "spearman_p":       round(spear_p,  4),
    "helpful_pct":      round(helpful_pct, 0),
    "mood_improvement": round(mood_mean, 1),
    "wilcoxon_p":       round(wilcox_p, 4),
}
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Updated {RESULTS_JSON}")
