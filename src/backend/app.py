# -*- coding: utf-8 -*-
import base64
import io
import logging
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap
import transformers
import xgboost as xgb
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from utils import preprocess_comment

# Configuration
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model("models/model.json")
    with open("models/model_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def load_bert_model():
    export_model_path = "models/bert_model"
    tokenizer = DistilBertTokenizer.from_pretrained(export_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(export_model_path)
    model.config.id2label = {0: "exploit_not_likely", 1: "exploit_likely"}
    model.config.label2id = {"exploit_not_likely": 0, "exploit_likely": 1}
    return model, tokenizer


df = pd.read_csv("data/scored_data.csv")
xgb_model, vectorizer = load_xgb_model()
bert_model, tokenizer = load_bert_model()
pred = transformers.pipeline(
    "text-classification", model=bert_model, tokenizer=tokenizer, return_all_scores=True
)
explainer = shap.Explainer(pred)


@app.route("/api/get_hashes_for_cve", methods=["POST"])
@cross_origin()
def get_hashes_for_cve():
    cve_id = request.json.get("cve_id")
    matching_entries = df[df["cveid"] == cve_id]
    hashes = matching_entries["hash"].tolist()
    return jsonify({"hashes": hashes})


@app.route("/api/get_suggestions", methods=["GET"])
@cross_origin()
def get_suggestions():
    suggestions = {
        "cve_ids": df["cveid"].unique().tolist(),
        "hashes": df["hash"].unique().tolist(),
    }
    return jsonify(suggestions)


@app.route("/api/explain", methods=["POST"])
@cross_origin()
def explain_prediction():
    cve_id = request.json.get("cve_id")
    hash = request.json.get("hash")

    entry = df[(df["cveid"] == cve_id) & (df["hash"] == hash)]

    # entry['comments'] = entry['comments'].apply(lambda x: ' '.join(eval(x)))
    logging.info(f"entry loaded {entry['comments']}")
    entry["comments"] = entry["comments"].apply(preprocess_comment)
    logging.info(f"entry preprocessed {entry['comments']}")

    logging.info(f"length of entry; {len(entry['comments'])}")
    data = pd.DataFrame({"text": entry["comments"], "emotion": entry["exploitability"]})
    shap_values = explainer(data["text"])
    logging.info("shap values loaded")

    # Generate the SHAP plot and save to an HTML string

    shap_html_text = shap.plots.text(shap_values, display=False)
    # Bar SHAP plot
    _, ax = plt.subplots(
        figsize=(8, 10)
    )  # Adjust the height for more space, especially if you have many features

    # Generate the SHAP bar plot
    shap.plots.bar(shap_values[0, :, 1], order=shap.Explanation.argsort, show=False)

    # Adjust tick label size and orientation for better readability (optional)
    ax.tick_params(
        axis="both", which="major", labelsize=10
    )  # Change '10' to adjust the font size
    for label in ax.get_xticklabels():
        label.set_ha("right")  # Horizontal alignment
        label.set_rotation(45)  # Rotation degree

    # Save the plot to a BytesIO object with a higher resolution
    buf = io.BytesIO()
    plt.tight_layout()  # Ensures that all labels fit in the figure
    plt.savefig(buf, format="png", dpi=300)
    logging.info("plot saved")
    plt.close()
    buf.seek(0)

    # Convert image to data URI
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    shap_html_bar = "data:image/png;base64," + image_base64
    # logging.info(shap_html_bar)
    logging.info("shap html loaded")

    # Get the prediction for the comment
    comment = entry["comments"].values[0]
    prediction = pred(comment)
    logging.info(f"prediction {prediction}")

    return jsonify(
        {
            "shap_plot_text": shap_html_text,
            "shap_plot_bar": shap_html_bar,
            "preds": prediction,
        }
    )


@app.route("/api/message", methods=["GET"])
@cross_origin()
def get_data():
    cveid_filter = request.args.get("cveid")

    # Selecting only 'cveid' and 'score' columns
    selected_df = df[["cveid", "score", "hash", "publication_date"]]

    if cveid_filter:
        filtered_df = selected_df[
            selected_df["cveid"].str.contains(cveid_filter, case=False)
        ]
    else:
        filtered_df = selected_df

    return jsonify(filtered_df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)
