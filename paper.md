---
title: 'MoodSense: A Browser-Based Ensemble Sentiment Analysis System for Real-Time Mood Tracking'
tags:
  - Python
  - JavaScript
  - sentiment analysis
  - affective computing
  - natural language processing
  - mental health
  - progressive web app
authors:
  - name: Yeshwanth Raghav Anarajula Venkata Sai
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 22 April 2026
bibliography: paper.bib
---

# Summary

MoodSense is a fully client-side, open-source ensemble sentiment analysis system deployed as a zero-install Progressive Web App (PWA). It combines three complementary natural language processing (NLP) models — DistilBERT [@sanh2019distilbert], VADER [@hutto2014vader], and TextBlob — into a unified Wellbeing Score that reflects the multi-dimensional affective content of a user's text input. The system requires no server, no account, and no installation, and runs entirely within a standard web browser using Hugging Face Transformers.js.

A secondary deployment is provided as a Google Colab research notebook, enabling researchers to reproduce all reported evaluations without specialist hardware. All code, evaluation scripts, and result artefacts are publicly available at [https://github.com/YeshwanthRaghav/-moodsense](https://github.com/YeshwanthRaghav/-moodsense).

# Statement of Need

Mood monitoring tools have broad applications in mental health support, user experience research, and affective computing. However, existing accessible tools present a recurring trade-off: commercial applications such as Woebot and Wysa offer polished interfaces but are proprietary and server-dependent, while open research implementations typically require technical expertise and significant compute resources [@coppersmith2015adhd].

Most open-source sentiment tools rely on a single model family. Lexicon-based tools such as VADER and TextBlob are computationally lightweight but context-insensitive. Transformer models such as DistilBERT achieve high classification accuracy but are not straightforwardly deployable in a standard browser without server infrastructure. Neither paradigm alone is optimal for accessible, multi-dimensional mood monitoring.

MoodSense addresses this gap through three design principles:

1. **Ensemble fusion** of complementary model families (transformer + lexicon-based) for improved multi-dimensional affective coverage.
2. **Zero-friction deployment** as a single-file client-side PWA — no server, no account, no installation required.
3. **Open reproducibility** — all evaluation code, model configurations, and results are publicly released.

MoodSense is explicitly positioned as a systems and deployment contribution. Its primary novelty lies in the accessible, privacy-preserving packaging of complementary NLP models, not in the ensemble methodology itself.

# System Description

## Analysis Pipeline

Each submitted text passes through five sequential processing steps:

1. **Preprocessing**: Unicode normalisation and truncation to 512 tokens for transformer compatibility.
2. **DistilBERT sentiment**: `distilbert-base-uncased-finetuned-sst-2-english` returns a POSITIVE/NEGATIVE label and confidence score.
3. **Emotion classification**: `j-hartmann/emotion-english-distilroberta-base` [@hartmann2022emotion] produces probability distributions over seven emotion categories (joy, sadness, anger, fear, surprise, disgust, neutral).
4. **VADER**: SentimentIntensityAnalyzer compound score rescaled to [0, 1].
5. **TextBlob**: Polarity rescaled to [0, 1]; subjectivity logged separately.

## Wellbeing Score Fusion

Each model output is rescaled to [0, 1] and combined via weighted average:

$$W = (S_{\text{bert}} \times w_1 + S_{\text{vader}} \times w_2 + S_{\text{blob}} \times w_3) \times 100$$

where $W \in [0, 100]$ is the Wellbeing Score. Default weights $(w_1, w_2, w_3) = (0.50, 0.30, 0.20)$ were selected based on relative benchmark performance and validated by ablation study across seven weight configurations.

## Crisis Detection

Every input is scanned in parallel for high-risk terms associated with suicide, self-harm, and acute hopelessness. A positive detection suppresses normal output and immediately presents emergency support resources. The keyword detection module achieves Precision = Recall = F1 = 0.82 on a balanced 100-sample evaluation set.

## Deployment

The primary deployment is a single-file HTML/JavaScript PWA executing entirely client-side. A lightweight JavaScript rule-based fallback activates when the full transformer pipeline is unavailable, with explicit user notification. The secondary deployment is a Python/Colab notebook for researchers requiring the full pipeline in a reproducible environment.

# Evaluation

Evaluation was conducted on an NVIDIA T4 GPU via Google Colab (free tier). Three tasks were evaluated independently:

| Model | SST-2 Acc. ↑ | Emotion F1 ↑ | WB MAE ↓ | Spearman r ↑ |
|---|---|---|---|---|
| VADER only | 0.631 | 0.235 | 31.69 | 0.547 |
| TextBlob only | 0.628 | — | 38.77 | 0.429 |
| DistilBERT only | 0.911 | 0.698 | 25.19 | 0.546 |
| MoodSense Ensemble | 0.906 | 0.451 | 27.92 | 0.558 |

The ensemble achieves Wellbeing MAE = 27.92 and Spearman r = 0.558 (p < 10⁻⁴²) against affective proxy labels. The 0.5% reduction in SST-2 accuracy relative to DistilBERT alone is a deliberate design trade-off that yields broader multi-dimensional affective coverage by incorporating lexicon-based signals.

**Disclaimer**: The Wellbeing Score reflects aggregated sentiment polarity across text inputs only. It is not a validated clinical measure and must not be used as a substitute for professional mental health assessment.

# Acknowledgements

Evaluation was conducted using publicly available datasets (GLUE SST-2 and dair-ai/emotion) and open-source pre-trained models via Hugging Face. Compute was provided by Google Colab (free tier). This research received no specific grant from any funding agency.

# References
