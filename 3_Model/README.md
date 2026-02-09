# Model Definition and Evaluation

## Model Selection

Beyond the LightGBM baseline, the project evaluated multiple sequence models:
- Temporal Fusion Transformer (TFT)
- iTransformer
- PatchTST
- Crossformer
- TimesNet

Goal: test whether deep temporal architectures can outperform the tabular baseline on sparse ICU trajectories.

## Feature Engineering / Input Variants

The same preprocessing families were compared across models:
- **Raw** input
- **Low preprocessing:** forward fill
- **High preprocessing:** forward fill + dynamic recency features

The high-preprocessing variant gave the best overall performance for both LightGBM and TFT.

## Hyperparameter Tuning

- Tuning was constrained by compute/time.
- For deep models, only limited sweeps were run (sequence length, learning rate, dropout/regularization, batch size, class weighting).
- Strong regularization (dropout, gradient clipping, lower learning rates) was used to reduce overfitting.

## Comparative Results (Utility Score, test)

- **LightGBM (high preprocessing): 0.4158**
- **Temporal Fusion Transformer (best run): 0.3988**
- **iTransformer: 0.3887**
- **PatchTST: 0.3756**
- **Crossformer: 0.3577**
- **TimesNet: 0.3332**

## Key Findings

- LightGBM remained the strongest model in this project setting.
- TFT was the best deep-learning alternative and came close to the baseline.
- More sophisticated transformer variants did not surpass LightGBM under current data volume and tuning budget.

## Discussion and Limitations

- Deep models likely underfit/overfit trade-offs due to limited tuning and mostly single-fold training.
- The dataset is highly imbalanced and sparse, favoring robust tree-based methods.
- Transformer models may become more competitive with:
  - larger data,
  - broader hyperparameter search,
  - multi-modal features,
  - stronger compute budget.