from flask import Flask, jsonify, request  # Add 'request' here
from flask_cors import CORS, cross_origin
import pandas as pd    
import pickle
import shap
import xgboost as xgb
import logging
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import shap
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import ast

def truncate_text(text, max_chars=500):
    return text[:max_chars]

def preprocess_comment(comment):
    comment_list = ast.literal_eval(comment)
    comment = ' '.join(comment_list)
    comment = re.sub(r'[^a-zA-Z0-9\s]', '', comment)  # Remove all non-alphanumeric characters except spaces
    return truncate_text(comment, 500)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

df = pd.read_csv("scored_data.csv")

# Load your DataFrame here
model = xgb.XGBClassifier()
model.load_model("model" + ".json")

with open("model_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# load the model and tokenizer
export_model_path = "bert_model"
tokenizer = DistilBertTokenizer.from_pretrained(export_model_path)
model = DistilBertForSequenceClassification.from_pretrained(export_model_path)
# Modify the class names
model.config.id2label = {0: "exploit_not_likely", 1: "exploit_likely"}
model.config.label2id = {"exploit_not_likely": 0, "exploit_likely": 1}

pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)
explainer = shap.Explainer(pred)

@app.route('/api/get_hashes_for_cve', methods=['POST'])
@cross_origin()
def get_hashes_for_cve():
    cve_id = request.json.get("cve_id")
    matching_entries = df[df['cveid'] == cve_id]
    hashes = matching_entries['hash'].tolist()
    return jsonify({"hashes": hashes})

@app.route('/api/get_suggestions', methods=['GET'])
@cross_origin()
def get_suggestions():
    suggestions = {
        "cve_ids": df['cveid'].unique().tolist(),
        "hashes": df['hash'].unique().tolist()
    }
    return jsonify(suggestions)

@app.route('/api/explain', methods=['POST'])
@cross_origin()
def explain_prediction():
    cve_id = request.json.get("cve_id")
    hash = request.json.get("hash")

    entry = df[df['cveid'] == cve_id]
    entry = df[df['hash'] == hash]
    

    # entry['comments'] = entry['comments'].apply(lambda x: ' '.join(eval(x)))
    entry['comments'] = entry['comments'].apply(preprocess_comment)
    logging.info("entry loaded", entry)
    logging.info(f"length of entry; {len(entry)}")
    data = pd.DataFrame({'text':entry['comments'],'emotion':entry['exploitability']})
    shap_values = explainer(data['text'][:1])
    logging.info("shap values loaded", shap_values)

    # Generate the SHAP plot and save to an HTML string
    shap_html = shap.plots.text(shap_values, display=False)
    # logging.info("shap html loaded", shap_html)
    # Return the HTML content as the response
    return jsonify({"shap_plot": shap_html})

@app.route('/api/message', methods=['GET'])
@cross_origin()
def get_data():
    cveid_filter = request.args.get('cveid')
    
    # Selecting only 'cveid' and 'score' columns
    selected_df = df[['cveid', 'score', 'hash', 'publication_date']]
    
    if cveid_filter:
        filtered_df = selected_df[selected_df['cveid'].str.contains(cveid_filter, case=False)]
    else:
        filtered_df = selected_df
    
    return jsonify(filtered_df.to_dict(orient='records'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)