# -*- coding: utf-8 -*-
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from transformers import (
    AdamW,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import preprocess_comment, preprocess_data

logging.basicConfig(level=logging.INFO)


def downsample_data(data):
    # Separate positive and negative samples
    positive_samples = data[data["exploitability"] == 1]
    negative_samples = data[data["exploitability"] == 0]

    # Determine the size of the smaller class
    minority_size = min(len(positive_samples), len(negative_samples))

    # Randomly select samples from both classes to get balanced data
    positive_downsampled = positive_samples.sample(n=minority_size, random_state=42)
    negative_downsampled = negative_samples.sample(n=minority_size, random_state=42)

    # Combine and shuffle
    downsampled_data = pd.concat([positive_downsampled, negative_downsampled])
    downsampled_data = downsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return downsampled_data


def train_and_save_bert_model(data_path: str, export_model_path: str):
    logging.info("Training BERT model...")
    # Load the data
    data = pd.read_csv(data_path)
    data = preprocess_data(data)
    data["comments"] = data["comments"].apply(lambda x: " ".join(eval(x)))
    data["comments"] = data["comments"].apply(preprocess_comment)
    # Downsample the data
    # data = downsample_data(data)

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    logging.info("Tokenizer loaded.")
    # Tokenize the dataset
    encoded_data = tokenizer(
        data["comments"].tolist(), padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )
    input_ids = encoded_data["input_ids"]
    attention_masks = encoded_data["attention_mask"]
    labels = torch.tensor(data["exploitability"].tolist())
    logging.info("Dataset tokenized.")

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    BATCH_SIZE = 128  # adjust as per your system's capabilities
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logging.info("Dataloaders constructed")
    # Load model

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to("cuda")  # if using GPU
    logging.info("Model loaded.")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 3  # assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    for epoch in range(3):  # adjust number of epochs as needed
        logging.info(f"Epoch {epoch + 1} of 3")
        model.train()
        logging.info("Training...")
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Validation
        model.eval()
        total_eval_accuracy = 0
        logging.info("Evaluating..")
        for batch in tqdm(validation_dataloader):
            with torch.no_grad():
                input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total_eval_accuracy += (predictions == labels).sum().item()

        avg_val_accuracy = total_eval_accuracy / val_size
        logging.info(f"Validation Accuracy for epoch {epoch}: {avg_val_accuracy:.2f}")

    # Save model
    model.save_pretrained(export_model_path)
    tokenizer.save_pretrained(export_model_path)
    logging.info("Model and tokenizer saved.")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_and_save_bert_model(data_path="data.csv", export_model_path="bert_model")
