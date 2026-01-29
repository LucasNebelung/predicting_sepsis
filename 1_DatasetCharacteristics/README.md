# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** (https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis/data)
- **Dataset Owner/Contact:** [Derived from the PhysioNet/Computing in Cardiology Challenge 2019 – “Early Prediction of Sepsis from Clinical Data”]

### Dataset Characteristics
- **Number of Observations:** [40,336 ICU stays/patients in the dataset with 1,552,210 individual datapoints; each stay is recorded as an hourly multivariate time series (ICULOS = hours since ICU admission)]
- **Number of Features:** 
[40 clinical predictor variables (vital signs, lab values, and demographics per hour) + time/demographic columns like ICULOS/HospAdmTime plus ID/label columns such as Patient_ID and SepsisLabel.]

### Target Variable/Label
- **Label Name:** SepsisLabel 
- **Label Type:** [Binary Classification]
- **Label Description:** [Indicates whether the patient is in the “sepsis window” at a given hour, following the PhysioNet Challenge definition. For sepsis patients, the label has been shifted 6 hours earlier than clinical onset, so 1 marks hours from t_sepsis − 6 h onward; all previous hours are 0. For non-sepsis patients, SepsisLabel is always 0]
- **Label Values:** [
0 – No sepsis at this hour
1 – Sepsis (or within 6 hours before sepsis onset) at this hour]
- **Label Distribution:** The dataset is highly imbalanced. Only a small minority of patients (≈7–8%) ever have SepsisLabel = 1 at any time; the vast majority never develop sepsis (all labels 0).

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**Vital signs** (columns 1–8 – per hour, continuous)

HR – Heart rate (beats per minute).

O2Sat – Peripheral oxygen saturation (%) via pulse oximetry.

Temp – Body temperature (°C).

SBP – Systolic blood pressure (mmHg).

MAP – Mean arterial pressure (mmHg).

DBP – Diastolic blood pressure (mmHg).

Resp – Respiratory rate (breaths per minute).

EtCO2 – End-tidal CO₂ (mmHg).

**Laboratory values** (columns 9–34 – per hour, continuous)

Metabolic, respiratory, liver, kidney, coagulation and hematologic labs, typically sparsely sampled and heavily missing:

BaseExcess – Excess bicarbonate (mmol/L).

HCO3 – Bicarbonate (mmol/L).

FiO2 – Fraction of inspired oxygen (%).

pH – Arterial blood pH (unitless).

PaCO2 – Arterial CO₂ partial pressure (mmHg).

SaO2 – Arterial oxygen saturation (%).

AST – Aspartate transaminase (IU/L).

BUN – Blood urea nitrogen (mg/dL).

Alkalinephos – Alkaline phosphatase (IU/L).

Calcium – Serum calcium (mg/dL).

Chloride – Serum chloride (mmol/L).

Creatinine – Serum creatinine (mg/dL).

Bilirubin_direct – Direct bilirubin (mg/dL).

Glucose – Serum glucose (mg/dL).

Lactate – Lactic acid (mg/dL).

Magnesium – Serum magnesium (mmol/dL).

Phosphate – Serum phosphate (mg/dL).

Potassium – Serum potassium (mmol/L).

Bilirubin_total – Total bilirubin (mg/dL).

TroponinI – Cardiac troponin I (ng/mL).

Hct – Hematocrit (%).

Hgb – Hemoglobin (g/dL).

PTT – Partial thromboplastin time (seconds).

WBC – White blood cell count (10³/µL).

Fibrinogen – Fibrinogen (mg/dL).

Platelets – Platelet count (10³/µL).

**Demographics & administrative/time features** (columns 35–40)

These are either static per ICU stay or change deterministically with time:

Age – Patient age in years (coded as 100 for age ≥ 90).

Gender – Encoded as 0 = female, 1 = male.

Unit1 – ICU type indicator (e.g. MICU = medical ICU); 0/1 flag.

Unit2 – ICU type indicator (e.g. SICU = surgical ICU); 0/1 flag.

HospAdmTime – Hours between hospital admission and ICU admission (may be negative).

ICULOS – ICU length of stay in hours since ICU admission (acts as the time index).

Patient_ID – Integer identifier for each ICU stay; used to group rows belonging to the same patient in the flattened CSV. 

SepsisLabel – Binary target label as described above (0/1).

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
