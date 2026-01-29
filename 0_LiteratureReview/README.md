# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: The Signature-Based Model for Early Detection of Sepsis

  - **[https://www.researchgate.net/publication/338627468\_The\_Signature-Based\_Model\_for\_Early\_Detection\_of\_Sepsis\_from\_Electronic\_Health\_Records\_in\_the\_Intensive\_Care\_Unit]()**

  - **Objective**:Early prediction of sepsis (≥6 h before onset) from ICU EHR time-series using the same PhysioNet/CinC 2019 challenge dataset.
  - **Methods**:Signature-based feature extraction from multivariate time series, combined with hand-crafted clinical features and a gradient-boosted model (LightGBM).
  - **Outcomes**:Winning submission (1st place) in the PhysioNet Computing in Cardiology Challenge 2019 Sepsis Prediction task.
  - **Relation to the Project**:Used as the baseline model to reproduce (signature features + gradient boosting) and extend with more advanced approaches.

- **Source 2**: [A time series driven model for early sepsis prediction based on transformer module]

  - **[https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-023-02138-6]()**
  - **Objective**:Early detection of sepis 4, 8, and 12h prior to sepsis diagnosis using the eICU research database (considerably larger database with 200000 patients from 208 differnet hospitals)
  - **Methods**:RNN, LSTM, CNN - Transformer, LSTM - Transformer models were attempted
  - **Outcomes**: According to authors, "the Transformer-based model demonstrated exceptional predictive capabilities, particularly within the earlier time window (i.e., 12 h before onset)" 
  - **Relation to the Project**: Goal would be replicate results

- **Source 3**: [Title of Source 3]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:
