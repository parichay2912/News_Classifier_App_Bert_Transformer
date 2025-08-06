import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from torch.optim import AdamW
from tqdm import tqdm
from etl import preprocess_data, get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# Define your model list
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]

def train_model(model_name, num_labels=4, epochs=1, batch_size=16):
    print(f"\n--- Training {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    df = preprocess_data("data/ag_news/train.csv")
    train_loader, val_loader = get_dataloaders(df, tokenizer, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{model_name}] Epoch {epoch+1} - Loss: {total_loss:.4f}")

    save_path = f"model/{model_name.replace('/', '_')}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved {model_name} to {save_path}")

if __name__ == "__main__":
    for model_name in model_names:
        train_model(model_name)
