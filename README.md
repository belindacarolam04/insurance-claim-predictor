# Health Insurance Claim Predictor

An end-to-end Machine Learning web application that predicts whether a health insurance claim will be approved or denied and estimates the claim amount for approved claims.

The project demonstrates a complete ML pipeline including data preprocessing, feature engineering, model training, evaluation, serialization, and deployment using Streamlit.

---

## Live Demo

🔗 **Streamlit App:**  
https://insurance-claim-predictor.streamlit.app

---

##  Project Overview

Insurance companies handle thousands of claims daily. Manually evaluating claim approval and estimating claim amounts can be time-consuming.

This project aims to build a machine learning solution that can:

- Predict whether a claim will be approved or rejected.
- Estimate the expected claim amount for approved claims.
- Provide quick predictions through an interactive web application.

The project uses two separate machine learning models:

1. **XGBoost Classifier**
   - Predicts claim approval status.

2. **XGBoost Regressor**
   - Predicts claim amount for approved claims.

---

#  Features

✅ Claim approval prediction  
✅ Claim amount estimation  
✅ Interactive Streamlit user interface  
✅ End-to-end ML pipeline  
✅ Saved trained models using Joblib  
✅ Real-time prediction  

---

#  Machine Learning Approach

## 1. Claim Approval Prediction (Classification)

The original claim amount was transformed into a binary target variable.

### Target Definition:

```
Claim Amount > 20000  → Approved (1)

Claim Amount <= 20000 → Not Approved (0)
```

An **XGBoost Classifier** was trained to classify claims.

### Model Performance:

| Metric | Score |
|--------|-------|
| Accuracy | 97.5% |
| Cross Validation Score | 96.2% |

---

## 2. Claim Amount Prediction (Regression)

Claim amount prediction was performed only on approved claims because denied claims do not represent actual payout values.

An **XGBoost Regressor** was trained to estimate the expected claim amount.

### Model Performance:

| Metric | Score |
|--------|-------|
| R² Score | 0.958 |

---

#  Machine Learning Workflow

```
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Streamlit Deployment
```

---

# Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### Deployment
- Streamlit

### Development Tools
- Jupyter Notebook
- Git & GitHub

---

#  Project Structure

```
insurance-claim-predictor/

│
├── app.py
│
├── healthinsurance.csv
│
├── xgb_classifier.pkl
│
├── xgb_regressor.pkl
│
├── classification_features.pkl
│
├── regression_features.pkl
│
├── requirements.txt
│
└── README.md
```

---

#  File Description

| File | Description |
|------|-------------|
| app.py | Streamlit application for prediction |
| healthinsurance.csv | Dataset used for training |
| xgb_classifier.pkl | Trained XGBoost classification model |
| xgb_regressor.pkl | Trained XGBoost regression model |
| classification_features.pkl | Saved features for classification model |
| regression_features.pkl | Saved features for regression model |
| requirements.txt | Required Python packages |

---

#  Dataset Features

The model uses health and demographic information:

| Feature | Description |
|---------|-------------|
| Age | Age of the customer |
| Weight | Weight of the customer |
| Blood Pressure | Blood pressure level |
| Gender | Gender information |
| City | Customer location |
| Job | Occupation |
| Disease | Health condition |

---

#  Installation & Usage

## Clone Repository

```bash
git clone https://github.com/belindacarolam04/insurance-claim-predictor.git
```

## Navigate to Project Directory

```bash
cd insurance-claim-predictor
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

#  Results

## Classification Model

**Algorithm:** XGBoost Classifier

Performance:

- Accuracy: 97.5%
- Cross Validation Score: 96.2%


## Regression Model

**Algorithm:** XGBoost Regressor

Performance:

- R² Score: 95.8%

---

#  Key Learnings

Through this project, I learned:

- How to build an end-to-end machine learning pipeline.
- Importance of defining the right ML problem.
- Handling classification and regression tasks together.
- Feature preprocessing for real-world datasets.
- Model evaluation and performance optimization.
- Deploying ML models as interactive web applications.

---

#  Future Improvements

- Add Explainable AI using SHAP values.
- Improve feature engineering.
- Add prediction history storage.
- Deploy using Docker and cloud platforms.
- Add model monitoring.
- Improve UI/UX design.

---

#  Author

**Belinda Carol**

Computer Science Student | Aspiring Data Scientist

GitHub:
https://github.com/belindacarolam04


---

 If you found this project useful, consider giving it a star!
