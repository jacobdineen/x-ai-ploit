# -*- coding: utf-8 -*-
import logging
import pickle

import pandas as pd
import xgboost as xgb
from utils import timer

# Initialize logging
logging.basicConfig(level=logging.INFO)


@timer
def main(input_data_path: str, model_path: str, vectorizer_path: str, out_path: str) -> pd.DataFrame:
    """
    Score input data using a saved model and vectorizer.

    Parameters:
    - input_data_path (str): Path to the CSV containing the input data.
    - model_path (str): Path to the saved XGBoost model.
    - vectorizer_path (str): Path to the saved TF-IDF vectorizer.

    Returns:
    - pd.DataFrame: Original data with an added 'score' column.
    """

    # Load the saved model and vectorizer
    model = xgb.XGBClassifier()
    model.load_model(model_path + ".json")
    with open(vectorizer_path + ".pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

    # Load and preprocess the input data
    data = pd.read_csv(input_data_path)
    data["strings"] = data["strings"].apply(lambda x: " ".join(eval(x)))

    # Transform the input features using the loaded vectorizer
    X = vectorizer.transform(data["strings"])

    # Predict scores using the loaded model
    scores = model.predict_proba(X)[:, 1]  # Assuming binary classification and you want the probability of class 1

    # Append the scores to the original input data
    data["score"] = scores
    data.to_csv(f"{out_path}.csv")
    logging.info(f"Scores saved to {out_path}.csv")
    return data
