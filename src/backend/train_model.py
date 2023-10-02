import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import logging
import warnings
from sklearn.utils import validation
from xgboost import data as xgboost_data
from utils import timer 


warnings.filterwarnings("ignore", category=FutureWarning, module=validation.__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module=xgboost_data.__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

def preprocess_data(data):
    # Convert 'True' to 1 and any other value (including NaN) to 0
    data['exploitability'] = data['exploitability'].apply(lambda x: 1 if x == True else 0)
    return data
@timer
def train_and_save_model(data_path: str, export_model_path: str):
    # Load the data
    data = pd.read_csv(data_path)

    # Preprocess data
    data = preprocess_data(data)
    # curl -X POST -H "Content-Type: application/json" -d '{"cve_id": "cve-2008-5499"}' http://localhost:3001/api/explain
    # If `strings` contains lists of strings, join them into a single string
    data['strings'] = data['strings'].apply(lambda x: ' '.join(eval(x)))

    # Split data into training and holdout sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        data['strings'], data['exploitability'], test_size=0.2, random_state=42)

    # Using TF-IDF to convert text data to numerical matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # Fit on training data and transform both training and holdout data
    X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
    X_holdout_transformed = tfidf_vectorizer.transform(X_holdout)

    # Initialize and train XGBoost model
    model = xgb.XGBClassifier()
    logging.info("Training model...")
    model.fit(X_train_transformed, y_train)

    # Predict on the holdout set
    y_pred = model.predict(X_holdout_transformed)
    accuracy = accuracy_score(y_holdout, y_pred)
    logging.info(f"Holdout set accuracy: {accuracy:.2f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_holdout, y_pred))
    logging.info("\nConfusion Matrix:")
    logging.info(confusion_matrix(y_holdout, y_pred))

    # Save the model and the vectorizer
    model.save_model(export_model_path + ".json")
    logging.info(f"Model saved to {export_model_path}")
    with open(export_model_path + "_vectorizer.pkl", 'wb') as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)

    logging.info("Model and vectorizer saved.")





def main(data_path: str, export_model_path: str):
    train_and_save_model(data_path, export_model_path)


if __name__ == "__main__":
    DATA_PATH = "data.csv"
    MODEL_PATH = "model.pkl"
    train_and_save_model(DATA_PATH, MODEL_PATH)
