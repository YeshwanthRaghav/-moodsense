"""
MoodSense Evaluation Script
============================
Run this in Google Colab (free tier, GPU not required — CPU is fine).

Steps:
  1. Open a new Colab notebook
  2. Paste this entire file into a code cell
  3. Click Runtime → Run all
  4. Copy the printed tables into your IEEE paper

Outputs:
  - TABLE II  : Model Performance Comparison (SST-2 Acc, Emotion F1, WB MAE, Spearman r)
  - TABLE III : Fusion Weight Ablation (SST-2 Accuracy)
  - moodsense_results.json : machine-readable results for user_study_analysis.py
"""

# ─────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("transformers")
pip("datasets")
pip("torch")
pip("textblob")
pip("vaderSentiment")
pip("scipy")
pip("scikit-learn")

print("✅ Dependencies installed. Now run Cell 2.")


# ─────────────────────────────────────────────
# CELL 2 — Load models & run SST-2 + Emotion
# ─────────────────────────────────────────────
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")

# ── Load models ──────────────────────────────
bert_pipe  = pipeline("text-classification",
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       device=device)
emo_pipe   = pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base",
                       device=device, top_k=None)
vader      = SentimentIntensityAnalyzer()
print("✅ Models loaded.")

# ── Helpers ──────────────────────────────────
def bert_score(text):
    r = bert_pipe(text[:512], truncation=True)[0]
    return r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]

def vader_score(text):
    c = vader.polarity_scores(text)["compound"]
    return (c + 1) / 2          # rescale [-1,1] → [0,1]

def blob_score(text):
    p = TextBlob(text).sentiment.polarity
    return (p + 1) / 2

def ensemble_score(text, w=(0.5, 0.3, 0.2)):
    return w[0]*bert_score(text) + w[1]*vader_score(text) + w[2]*blob_score(text)

def to_binary(score):
    return 1 if score >= 0.5 else 0

# ── Experiment 1: SST-2 ──────────────────────
print("\nRunning SST-2...")
sst = load_dataset("glue", "sst2", split="validation")
texts  = sst["sentence"]
labels = sst["label"]          # 1=positive, 0=negative

vader_preds  = [to_binary(vader_score(t))    for t in texts]
blob_preds   = [to_binary(blob_score(t))     for t in texts]
bert_preds   = [to_binary(bert_score(t))     for t in texts]
ens_preds    = [to_binary(ensemble_score(t)) for t in texts]

exp1 = {
    "VADER":      round(accuracy_score(labels, vader_preds), 4),
    "TextBlob":   round(accuracy_score(labels, blob_preds),  4),
    "DistilBERT": round(accuracy_score(labels, bert_preds),  4),
    "Ensemble":   round(accuracy_score(labels, ens_preds),   4),
}
print("✅ SST-2:", exp1)

# ── Experiment 2: Emotion F1 ─────────────────
print("\nRunning Emotion...")
EMO_MAP = {
    "joy":"joy","love":"joy","optimism":"joy",
    "sadness":"sadness","grief":"sadness",
    "anger":"anger","disgust":"anger",
    "fear":"fear","nervousness":"fear",
    "surprise":"surprise","confusion":"surprise",
    "neutral":"neutral","approval":"neutral","disapproval":"neutral",
}
emo_ds  = load_dataset("dair-ai/emotion", split="test")
emo_texts = emo_ds["text"]
emo_true  = emo_ds["label"]
emo_names = emo_ds.features["label"].names   # e.g. ['sadness','joy',...]

# Hartmann predictions → map to dair-ai label indices
hartmann_raw = emo_pipe(emo_texts, truncation=True, max_length=512, batch_size=32)
def top_label(result):
    return max(result, key=lambda x: x["score"])["label"].lower()

hart_labels_str = [top_label(r) for r in hartmann_raw]
# Map Hartmann emotion → dair-ai emotion name → dair-ai index
HART_TO_DAIRAI = {
    "joy":"joy","sadness":"sadness","anger":"anger",
    "fear":"fear","surprise":"surprise","disgust":"disgust","neutral":"neutral",
}
# dair-ai index lookup
name_to_idx = {n: i for i, n in enumerate(emo_names)}
emo_preds = [name_to_idx.get(HART_TO_DAIRAI.get(l, "neutral"), 0) for l in hart_labels_str]

# VADER emotion baseline (map compound → positive/negative only, crude)
vader_emo = [name_to_idx.get("joy" if vader_score(t) >= 0.5 else "sadness", 0) for t in emo_texts]

exp2 = {
    "Hartmann_F1":  round(f1_score(emo_true, emo_preds,  average="macro", zero_division=0), 4),
    "Hartmann_acc": round(accuracy_score(emo_true, emo_preds), 4),
    "VADER_F1":     round(f1_score(emo_true, vader_emo,   average="macro", zero_division=0), 4),
}
print("✅ Emotion:", exp2)

print("\n✅ Cell 2 done. Now run Cell 3.")


# ─────────────────────────────────────────────
# CELL 3 — Wellbeing Score correlation & Ablation
# ─────────────────────────────────────────────

# ── Experiment 3: Wellbeing Score vs proxy ───
VALENCE = {"joy":85,"love":80,"surprise":65,"sadness":20,"anger":15,"fear":25,"neutral":50}
proxy   = [VALENCE.get(emo_names[l], 50) for l in emo_true[:500]]
wb_texts = emo_texts[:500]

wb_scores = [ensemble_score(t) * 100 for t in wb_texts]
vader_wb  = [vader_score(t) * 100     for t in wb_texts]

spear_ens,   _ = spearmanr(wb_scores, proxy)
spear_vader, _ = spearmanr(vader_wb,  proxy)
mae_ens   = float(np.mean(np.abs(np.array(wb_scores)  - np.array(proxy))))
mae_vader = float(np.mean(np.abs(np.array(vader_wb)   - np.array(proxy))))

exp3 = {
    "Ensemble_r":   round(spear_ens,   3),
    "Ensemble_MAE": round(mae_ens,     1),
    "VADER_r":      round(spear_vader, 3),
    "VADER_MAE":    round(mae_vader,   1),
}
print("Wellbeing:", exp3)

# ── Experiment 4: Ablation ───────────────────
ablation_configs = [
    ("BERT only",         (1.0, 0.0, 0.0)),
    ("VADER only",        (0.0, 1.0, 0.0)),
    ("TextBlob only",     (0.0, 0.0, 1.0)),
    ("MoodSense default", (0.5, 0.3, 0.2)),
    ("BERT-heavy",        (0.6, 0.2, 0.2)),
    ("Equal B+V",         (0.4, 0.4, 0.2)),
    ("Equal",             (0.33,0.33,0.34)),
]

ablation = []
for name, w in ablation_configs:
    preds = [to_binary(ensemble_score(t, w)) for t in texts]
    acc   = round(accuracy_score(labels, preds), 3)
    ablation.append({"Config": name, "Weights": f"{w[0]}/{w[1]}/{w[2]}", "Acc": acc})

# ── Print tables ─────────────────────────────
SEP = "=" * 65

print(f"\n{SEP}")
print("TABLE II — PASTE INTO PAPER")
print(SEP)
print(f"{'Model':<22} {'SST-2':>6} {'EmoF1':>6} {'MAE':>5} {'Spear-r':>8}")
print("-" * 50)
rows = [
    ("VADER only [2]",      exp1["VADER"],       exp2["VADER_F1"],  exp3["VADER_MAE"],  exp3["VADER_r"]),
    ("TextBlob only",       exp1["TextBlob"],     "—",               "—",                "—"),
    ("DistilBERT only [1]", exp1["DistilBERT"],   exp2["Hartmann_F1"], exp3["Ensemble_MAE"], exp3["Ensemble_r"]),
    ("MoodSense Ensemble",  exp1["Ensemble"],     exp2["Hartmann_F1"], exp3["Ensemble_MAE"], exp3["Ensemble_r"]),
]
for r in rows:
    name, sst, emo, mae, sp = r
    print(f"{name:<22} {str(sst):>6} {str(emo):>6} {str(mae):>5} {str(sp):>8}")

print(f"\n{SEP}")
print("TABLE III — PASTE INTO PAPER (Ablation)")
print(SEP)
print(f"{'Configuration':<22} {'Weights':>14} {'SST-2 Acc':>10}")
print("-" * 50)
for row in ablation:
    print(f"  {row['Config']:<20} {row['Weights']:>14} {row['Acc']:>10}")

print(f"\n✅ ALL DONE — paste the tables above here!")

# ── Save JSON for user_study_analysis.py ─────
results = {
    "experiment_1_sst2":      exp1,
    "experiment_2_emotion":   exp2,
    "experiment_3_wellbeing": exp3,
    "experiment_4_ablation":  ablation,
}
with open("moodsense_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved moodsense_results.json")
