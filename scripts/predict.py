# scripts/predict.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import os

# Load model and tokenizer
MODEL_PATH = "model/saved_model/"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def predict_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]

def predict_csv(input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if not {'title', 'description'}.issubset(df.columns):
        print("Input CSV must contain 'title' and 'description' columns.")
        return

    texts = df['title'] + " " + df['description']
    predictions = []

    for text in tqdm(texts, desc="Predicting"):
        pred = predict_text(text)
        predictions.append(pred)

    df['predicted_category'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT News Classifier - Predict")
    parser.add_argument("--text", type=str, help="Single text input: 'title description'")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to output CSV file")

    args = parser.parse_args()

    if args.text:
        print("Predicted Category:", predict_text(args.text))
    elif args.input_csv:
        predict_csv(args.input_csv, args.output_csv)
    else:
        print("Please provide either --text or --input_csv")
