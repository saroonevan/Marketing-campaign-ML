# 📊 Marketing Campaign Performance Predictor

> An end-to-end Machine Learning project that predicts **Revenue** and **Profit/Loss** for marketing campaigns across three Indian beauty brands — **Nykaa**, **Purplle**, and **Tira**.

---

## 🎯 Project Overview

This project builds a complete ML pipeline — from raw data to a deployed Streamlit web application — that helps marketing teams:

- 💰 **Predict Revenue** a campaign will generate (before it runs)
- 📈 **Predict Profit or Loss** outcome of any campaign
- 📊 **Explore EDA insights** across brands, channels, audiences, and time

---

## 🖥️ Live App Preview

| Home Page | Revenue Prediction | Profit/Loss Prediction |
|:---------:|:-----------------:|:---------------------:|
| Model summary & KPIs | Input form + Gauge chart | Input form + Donut chart |

**Run locally:**
```bash
streamlit run app/App.py
```

---

## 📁 Project Structure

```
marketing_campaign_ml/
│
├── app/
│   └── App.py                          # Streamlit web application
│
├── models/
│   ├── regression_model.pkl            # XGBoost Regressor (R² 99.20%)
│   └── classification_model.pkl        # Logistic Regression + SMOTE (97.28%)
│
├── Data.ipynb                          # Complete ML pipeline notebook
│
├── cleaned_combined_data.csv           # Cleaned dataset (166,665 rows)
│
├── nykaa_campaign_data_with_nulls.csv  # Raw Nykaa data
├── purplle_campaign_data_with_nulls.csv# Raw Purplle data
├── tira_campaign_data_with_nulls.csv   # Raw Tira data
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 📊 Dataset

| Property        | Details                          |
|----------------|----------------------------------|
| Total Campaigns | 166,665                         |
| Brands          | Nykaa, Purplle, Tira            |
| Date Range      | July 2024 – June 2025           |
| Raw Features    | 17 columns                      |
| Encoded Features| 40 columns                      |
| Class Balance   | Profit 95.01% / Loss 4.99%      |

### Original Columns
```
Brand, Campaign_Type, Target_Audience, Duration, Channel_Used,
Impressions, Clicks, Leads, Conversions, Revenue,
Acquisition_Cost, Engagement_Score, Language, Customer_Segment,
Date, Campaign_ID, Updated_ROI
```

---

## 🔄 ML Pipeline

```
Raw CSV Files (3 brands)
        ↓
Data Loading & Merging
        ↓
Data Cleaning
  → Null handling (Revenue=0, Cost=mean, Language=Native)
  → Date parsing & formatting
  → Campaign ID generation
  → Outlier capping (Winsorization 1st-99th percentile)
        ↓
Feature Engineering
  → Channel one-hot encoding (Email, Facebook, Google,
    Instagram, WhatsApp, YouTube)
  → Updated_ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost
  → Profit_Flag = (Updated_ROI > 0)
  → Month_Name column
        ↓
Exploratory Data Analysis (EDA)
  → Brand revenue comparison
  → Campaign type performance
  → Channel analysis
  → Audience segmentation
  → Monthly spend trends
        ↓
Encoding (pd.get_dummies — drop_first=False)
  → Brand, Campaign_Type, Target_Audience,
    Language, Customer_Segment
        ↓
Model Training
  → 6 Regression models
  → 10 Classification models + SMOTE
        ↓
Model Saving (Pickle)
        ↓
Streamlit Web App
```

---

## 🤖 Models

### Regression — Predicting Revenue (₹)

| Model            | R² Score | RMSE      | Gap    |
|-----------------|----------|-----------|--------|
| Linear Regression| 77.11%   | ₹222,882  | 0.0038 |
| Ridge Regression | 77.11%   | ₹222,882  | 0.0038 |
| Lasso Regression | 77.11%   | ₹222,882  | 0.0038 |
| Decision Tree    | 99.21%   | ₹41,483   | 0.0079 |
| Random Forest    | 99.65%   | ₹27,550   | 0.0029 |
| **XGBoost** ✅   | **99.20%** | **₹41,744** | **0.0057** |

**Winner: XGBoost Regressor**
- R² Score: **99.20%**
- RMSE: **₹41,744**
- MAE: **₹12,191**
- Trained on: 133,332 rows

---

### Classification — Predicting Profit / Loss

| Model              | Accuracy | F1     | AUC    | Loss Recall |
|-------------------|----------|--------|--------|-------------|
| Logistic Reg.      | 98.20%   | 99.06% | 0.983  | 63.86%      |
| Decision Tree      | 100.00%  | 100.00%| 1.000  | 100.00%     |
| Random Forest      | 100.00%  | 100.00%| 1.000  | 100.00%     |
| XGBoost            | 99.90%   | 99.95% | 1.000  | 98.50%      |
| KNN                | 94.86%   | 97.36% | 0.514  | 0.00%       |
| Naive Bayes        | 99.99%   | 100.00%| 1.000  | 100.00%     |
| Gradient Boosting  | 100.00%  | 100.00%| 1.000  | 100.00%     |
| AdaBoost           | 100.00%  | 100.00%| 1.000  | 100.00%     |
| SVM                | 99.29%   | 99.63% | 0.994  | 87.31%      |
| LightGBM           | 100.00%  | 100.00%| 1.000  | 100.00%     |

**After SMOTE (balanced training):**

| Model              | Accuracy | F1     | Loss Recall |
|-------------------|----------|--------|-------------|
| **Logistic Reg.** ✅| **97.28%** | **98.55%** | **97.11%** |
| XGBoost            | 99.92%   | 99.96% | 99.04%      |

**Winner: Logistic Regression + SMOTE** *(Mentor approved)*
- Accuracy: **97.28%**
- F1 Score: **98.55%**
- ROC-AUC: **0.9973**
- Loss Recall: **97.11%**
- Trained on: **253,360 rows** (SMOTE balanced)

---

## ⚖️ SMOTE — Handling Imbalanced Data

```
Before SMOTE:
  Profit (1) → 126,680  (95%) ← majority
  Loss   (0) →   6,652  (5%)  ← minority

After SMOTE:
  Profit (1) → 126,680  (50%) ← balanced ✅
  Loss   (0) → 126,680  (50%) ← balanced ✅

Loss Recall improved:
  Before SMOTE → 63.86% (missing 36% of losses)
  After  SMOTE → 97.11% (catching 97% of losses) ✅
```

> SMOTE implemented manually using `NearestNeighbors` due to 
> compatibility issues between `imbalanced-learn 0.13.0` and `scikit-learn 1.9.0`

---

## 📱 Streamlit App Features

### Single Page Layout (Option C)
```
📋 Campaign Details
  ┌──────────────┬─────────────────┬──────────────┐
  │ Brand &      │ Performance     │ Channels     │
  │ Campaign     │ Numbers         │ ✅ Instagram  │
  └──────────────┴─────────────────┴──────────────┘

🔮 Predict Revenue AND Profit/Loss  [button]

🎯 Prediction Results
  ┌──────────────────┬────────────────────────┐
  │ 💰 Revenue       │ 📈 Profit / Loss        │
  │ ₹9,95,747        │ ✅ PROFITABLE 87%       │
  └──────────────────┴────────────────────────┘

📊 Visual Summary
  ┌──────────────────┬────────────────────────┐
  │ Revenue Gauge    │ Profit/Loss Donut      │
  └──────────────────┴────────────────────────┘

💡 Campaign Insight → Strong / Moderate / Weak
```

### Input Features (User Provides)
| Feature | Type | Options |
|--------|------|---------|
| Brand | Dropdown | Nykaa, Purplle, Tira |
| Campaign Type | Dropdown | Social Media, Email, SEO, Influencer, Paid Ads |
| Target Audience | Dropdown | College Students, Youth, Working Women, Premium Shoppers, Tier 2 City |
| Language | Dropdown | Hindi, English, Tamil, Bengali, Native |
| Customer Segment | Dropdown | 6 options |
| Duration | Slider | 5–30 days |
| Impressions | Number | 10,000–100,000 |
| Clicks | Number | 200–15,000 |
| Leads | Number | 50–9,000 |
| Conversions | Number | 10–7,000 |
| Acquisition Cost | Number | ₹8–₹15,000 |
| Engagement Score | Slider | 2.5–31.0 |
| Channels | Checkboxes | Email, Facebook, Google, Instagram, WhatsApp, YouTube |

---

## 🔑 Key Findings from EDA

```
Brand Analysis:
  Nykaa   → ₹27.20B  (33.4%) ← highest revenue
  Purplle → ₹27.11B  (33.3%)
  Tira    → ₹27.06B  (33.3%)
  → All brands nearly equal — balanced dataset ✅

Campaign Type:
  Influencer   → ₹19.54B ← 26% more than others!
  Paid Ads     → ₹15.56B
  Email        → ₹15.55B

Channel Performance:
  Instagram    → ₹490,927 avg ← highest
  YouTube      → ₹486,195 avg ← lowest
  (All channels similar — ~₹487K avg)

Audience:
  Premium Shoppers → ₹15.12M total spend ← most valuable
  College Students → ₹11.83M ← lowest

Monthly Trends:
  Peak   → Dec'24, Oct'24 (festive seasons) 🪔
  Lowest → Jun'25 (incomplete month)
```

---

## 🚀 Setup & Installation

### Prerequisites
```
Python 3.9+
pip
```

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/marketing-campaign-ml.git
cd marketing-campaign-ml
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app/App.py
```

### Step 5 — Open browser
```
http://localhost:8501
```

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn==1.8.0
xgboost
lightgbm
streamlit
plotly
matplotlib
seaborn
pickle5
```

---

## 📂 Model Artifacts

Each `.pkl` file contains a dictionary with:

```python
{
    'model'       : trained_model_object,
    'scaler'      : StandardScaler_object,
    'features'    : ['col1', 'col2', ...],  # 38 columns
    'model_name'  : 'XGBoost Regressor',
    'r2_score'    : 0.9920,                 # regression only
    'rmse'        : 41744.25,               # regression only
    'accuracy'    : 0.9728,                 # classification only
    'loss_recall' : 0.9711,                 # classification only
    'smote_used'  : True,
    'trained_on'  : '133,332 rows'
}
```

**Load a model:**
```python
import pickle

with open('models/regression_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

model   = pkg['model']
scaler  = pkg['scaler']
features= pkg['features']

# Predict
prediction = model.predict(X_input)
```

---

## 🧠 What I Learned

```
✅ End-to-end ML pipeline from raw CSV to deployed app
✅ Data cleaning — nulls, outliers, encoding
✅ Exploratory Data Analysis with Plotly and Seaborn
✅ Feature engineering — one-hot encoding, ROI calculation
✅ Regression models — Linear, Ridge, Lasso, DT, RF, XGBoost
✅ Classification models — 10 algorithms compared
✅ Imbalanced data handling — SMOTE technique
✅ Data leakage detection and prevention
✅ Model persistence with Pickle
✅ Streamlit web app deployment
✅ Plotly interactive visualisations
```

---

## 📌 Data Leakage — Key Decision

> **Updated_ROI** was identified as a leakage feature for classification since 
> `Profit_Flag = (Updated_ROI > 0)` — it directly encodes the target label.
> After mentor consultation, Updated_ROI was retained in regression (where it's a 
> legitimate feature) and handled appropriately in the classification pipeline.

---

## 👤 Author

**Saroon**  
B.Tech — Data Science & Artificial Intelligence  
SRM Institute of Science and Technology

---

## 📄 License

This project is created for educational purposes as part of a supervised ML course project.

---

## 🙏 Acknowledgements

- Mentor guidance throughout the project
- Dataset: Synthetic Indian beauty brand marketing data
- Libraries: scikit-learn, XGBoost, Streamlit, Plotly, Pandas

---

*Built with ❤️ using Python, scikit-learn, XGBoost, and Streamlit*
