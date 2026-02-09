# Evaluation

This folder contains the evaluation workflow used to compare all models under the **official PhysioNet 2019 protocol**.

## Evaluation Objective

Model outputs are hourly predicted probabilities per patient. These probabilities are converted into binary alarms via thresholding, then scored with the official challenge metrics.

The central objective is to optimize **clinical utility score**, not only accuracy.

## Official Metric

`evaluate_sepsis_score.py` (official PhysioNet script) reports:
- AUROC
- AUPRC
- Accuracy
- F-measure
- **Utility** (time-aware clinical utility)

Utility rewards early true-positive alerts and penalizes late/missed detections and false alarms with challenge-specific weighting.

## Threshold Sweep Procedure

A standardized threshold sweep is run on `train_thresh` predictions:
1. Sweep candidate thresholds
2. Evaluate each threshold with official utility
3. Select best threshold
4. Re-evaluate on held-out test predictions

This ensures all models are compared at their best operating point under the same protocol.

## Summary of Final Test Utility

- LightGBM (high preprocessing): **0.4158**
- Temporal Fusion Transformer: **0.3978–0.3988** (depending on run)
- iTransformer: **0.3887**
- PatchTST: **0.3756**
- Crossformer: **0.3577**
- TimesNet: **0.3332**
- qSOFA proxy baselines: **0.0279** and **0.0923** (aggressive alarm policy variant)

## Folder Contents

- `evaluate_sepsis_score.py`: official evaluator
- `Threshold_Sweep_And_Official_Eval.ipynb`: threshold optimization + final scoring
- `Predictions/`: model-specific predictions and final metrics JSON files

## Practical Interpretation

A utility around ~0.4 indicates meaningful clinical value compared with simple rule-based approaches in this setup. In this project, LightGBM provides the best utility-speed-robustness trade-off.