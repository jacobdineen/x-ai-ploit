from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline, pipeline
import pandas as pd
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

def extract_label1_score(prediction):
    for entry in prediction:
        if entry['label'] == 'LABEL_1':
            return entry['score']
    return None  # Return None if 'LABEL_1' is not found in prediction


def truncate_text(text, max_chars=500):
    return text[:max_chars]

def main():
    # Load the tokenizer and model
    export_model_path = "bert_model"
    tokenizer = DistilBertTokenizer.from_pretrained(export_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(export_model_path)
    logging.info("model and tokenizer loaded")

    # Setup the classification pipeline
    classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1, return_all_scores=True)

    # Load the data
    data = pd.read_csv("data.csv")
    data = data[:100]
    data['comments'] = data['comments'].apply(lambda x: ' '.join(eval(x)))
    logging.info("data loaded")


    max_chars = 500
    data['comments'] = data['comments'].apply(lambda x: truncate_text(x, max_chars))
    logging.info("all comments truncated")

    batch_size = 1028  # Adjust based on memory availability
    num_comments = len(data['comments'])
    all_predictions = []

    for i in tqdm(range(0, num_comments, batch_size)):
        batch_comments = data['comments'][i:i+batch_size].tolist()
        batch_predictions = classification_pipeline(batch_comments)
        all_predictions.extend(batch_predictions)


    all_label1_scores = [extract_label1_score(p) for p in all_predictions]
    print(all_label1_scores)

if __name__ == "__main__":
    main()