# Churn Prediction Model with SHAP Explainability

This project demonstrates a complete process for building, explaining, and serving a machine learning model for churn prediction using SHAP values. The model predicts customer churn for a telecommunications company based on various customer attributes and service usage patterns.

## 📊 Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. This dataset contains information about 7043 customers, including their demographics, services subscribed to, and whether they churned or not.

### Features

The dataset includes the following key features:

**Demographic Information:**
- Gender
- Senior Citizen status
- Partner status
- Dependents

**Service Information:**
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

**Contract Details:**
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Tenure (months)

**Target Variable:**
- Churn (Yes/No)

## 📦 Project Structure

- `exp.ipynb` – Model training, EDA, and SHAP explainer creation
- `client.ipynb` – Loads the trained model and explains individual predictions using SHAP
- `model.pkl` – Serialized Random Forest Classifier
- `data/churn.csv` – Dataset file

## 🛠️ Setup and Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn shap
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data` directory
4. Run the notebooks in order:
   - First `exp.ipynb` to train the model
   - Then `client.ipynb` to make predictions and get explanations

## 🔍 Model Details

The project uses a Random Forest Classifier for churn prediction, with SHAP (SHapley Additive exPlanations) values to explain the model's predictions. The model helps understand:
- Which features are most important for predicting churn
- How each feature contributes to individual predictions
- What factors are driving customer churn

## 📈 Results

The model provides both predictive power and interpretability, allowing business stakeholders to:
- Identify at-risk customers
- Understand the key factors contributing to churn
- Make data-driven decisions to reduce customer churn

## 🧪 Running Tests

Install the development dependencies and run the test suite with `pytest`:

```bash
pip install -e .[dev]
pytest
```

---

