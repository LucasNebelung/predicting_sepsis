# Baseline Model

**[LightGBM Notebook](LIGHTGBM.ipynb)**
**[Clinical Baseline (qSOFA proxy) Notebook](qSOFA.ipynb)**

## Baseline Model Results

### Model Selection
 **Baseline model type:** LightGBM (gradient-boosted decision trees)
 **Clinical comparator:** partial qSOFA proxy (rule-based, non-ML)
 **Rationale:**
    LightGBM is robust on sparse/imbalanced tabular clinical data
    Fast to train and easy to iterate on
    Strong prior evidence in PhysioNet 2019 sepsis work]

### Model Performance (Official PhysioNet Evaluator)
**Primary metric:** Utility score (clinical-timing-aware)
**Best LightGBM baseline:**
  Test Utility: **0.4158**
  AUROC: **0.8535**
  AUPRC: **0.1353**
  Accuracy: **0.7849**
  F1: **0.1114**
**qSOFA proxy baseline (for reference):**
  Test Utility: **0.0279** (threshold-optimized proxy)
  Aggressive alarm policy variant: **0.0923** utility

### Cross-Validation / Validation Strategy
-Patient-level split to prevent leakage.
-80/20 train-test split, plus a 5% patient-level holdout from training (`train_thresh`) for threshold optimization.
-LightGBM was tuned with stronger validation practices than deep models (which mostly used single-fold settings in this project stage).

### Evaluation Methodology
- **Data split:** Patient-wise split (`train_fit`, `train_thresh`, `test`)
- **Thresholding:** Standardized threshold sweep on `train_thresh` to maximize utility
- **Metrics:** AUROC, AUPRC, Accuracy, F1, and official Utility

### Metric Practical Relevance
- **Utility score** is the most clinically relevant metric here because it rewards *timely* detection and penalizes late/missed detection.
- **AUROC/AUPRC** summarize ranking quality under imbalance, while **F1/Accuracy** provide threshold-dependent operating-point context.
- In this application, a model with better utility is preferred even when accuracy is lower, because clinical timing dominates outcome value.

## Next Steps
This baseline serves as the reference for transformer-based and time-series deep-learning models in [3_Model/Time-Series-Library](../3_Model/README.md).


