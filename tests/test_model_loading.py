import pickle
import pandas as pd


def test_model_prediction_probability_within_range():
    with open('model/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    sample = pd.DataFrame([
        {
            'SeniorCitizen': 0,
            'Partner': 1,
            'Dependents': 1,
            'tenure': -0.7564030091239073,
            'OnlineSecurity': 0,
            'OnlineBackup': 0,
            'TechSupport': 0,
            'StreamingMovies': 1,
            'PaperlessBilling': 1,
            'MonthlyCharges': 0.739078922339588,
            'InternetService_Fiber optic': 1.0,
            'InternetService_No': 0.0,
            'PaymentMethod_Electronic check': 1.0,
            'Contract_One year': 0.0,
            'Contract_Two year': 0.0,
        }
    ])

    proba = float(loaded_model.predict_proba(sample)[0, 1])
    assert 0.0 <= proba <= 1.0
