# scripts/etl.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import re
import string

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['label'] = df['Class Index'] - 1  # labels: 0–3
    # Combine title and description, handle missing
    df['Title'] = df['Title'].fillna('')
    df['Description'] = df['Description'].fillna('')
    df['text'] = (df['Title'] + " " + df['Description']).apply(preprocess_text)
    return df[['text', 'label']]

def tokenize_data(df, tokenizer, max_length=128):
    return tokenizer(
        list(df['text']),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def get_dataloaders(df, tokenizer, batch_size=16):
    from torch.utils.data import DataLoader, TensorDataset

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    train_tokens = tokenize_data(train_df, tokenizer)
    val_tokens = tokenize_data(val_df, tokenizer)

    train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], torch.tensor(train_df['label'].values))
    val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'], torch.tensor(val_df['label'].values))

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size)

if __name__ == "__main__":
    df = preprocess_data("data/ag_news.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader = get_dataloaders(df, tokenizer)
