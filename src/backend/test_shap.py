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
import io
import base64

logging.basicConfig(level=logging.INFO)


def truncate_text(text, max_chars=500):
    return text[:max_chars]

def preprocess_comment(comment):
    comment_list = ast.literal_eval(comment)
    comment = ' '.join(comment_list)
    comment = re.sub(r'[^a-zA-Z0-9\s]', '', comment)  # Remove all non-alphanumeric characters except spaces
    return truncate_text(comment, 500)

# load the model and tokenizer
export_model_path = "bert_model"
tokenizer = DistilBertTokenizer.from_pretrained(export_model_path)
model = DistilBertForSequenceClassification.from_pretrained(export_model_path)
# Modify the class names
model.config.id2label = {0: "exploit_not_likely", 1: "exploit_likely"}
model.config.label2id = {"exploit_not_likely": 0, "exploit_likely": 1}

pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)
explainer = shap.Explainer(pred)

df = pd.read_csv("scored_data.csv")
# print(df[df['exploitability'] == 1].head(20))

cve_id = "cve-2015-8103"
hash = "5b4fbac60182b69e9a0417ea726276c87c101335d6fbdc5e8c9a9a429235c655"
entry = df[(df['cveid'] == cve_id) & (df['hash'] == hash)]
print(entry['comments'].values)

# # entry['comments'] = entry['comments'].apply(lambda x: ' '.join(eval(x)))
# entry['comments'] = entry['comments'].apply(preprocess_comment)
# comment = entry["comments"].values[0]  # Assuming this results in a single string

# print(pred(entry["comments"].values[0]))

# print(entry['comments'])
# logging.info("entry loaded", entry)
# logging.info(f"length of entry; {len(entry)}")
# data = pd.DataFrame({'text':entry['comments'],'emotion':entry['exploitability']})
# shap_values = explainer(data['text'])
# logging.info("shap values loaded", shap_values)

# # Generate the SHAP plot and save to an HTML string
# # shap_html = shap.plots.text(shap_values, display=False)
# # Create a new figure with a specific size
# fig, ax = plt.subplots(figsize=(6, 4))  # You can adjust the figsize values as needed

# # Draw your SHAP plot
# shap.plots.bar(shap_values[0, :, 1], order=shap.Explanation.argsort, show=False)


# # Save the plot to a BytesIO object with a higher resolution
# buf = io.BytesIO()
# plt.savefig(buf, format="png", dpi=300)  # Increase dpi for higher resolution
# logging.info("plot saved")
# plt.close()
# buf.seek(0)

# # Convert image to data URI
# image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
# data_uri = "data:image/png;base64," + image_base64
# logging.info(f"data uri: {data_uri}")