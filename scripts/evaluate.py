import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(batch_size=32):
    tokenizer = BertTokenizer.from_pretrained("model/saved_model/")
    model = BertForSequenceClassification.from_pretrained("model/saved_model/")
    model.to(device)
    model.eval()

    df = pd.read_csv("smart_news_classifier/data/ag_news/test.csv")
    texts = (df['Title'].fillna('') + " " + df['Description'].fillna('')).tolist()
    labels = (df['Class Index']-1).tolist()

    preds = []
    true_labels = []

    # Tokenize all texts without tensors first (list of dicts)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    # Convert encodings to tensors
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels_tensor = torch.tensor(labels)

    # Create dataset and dataloader for batching
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch, labels_batch = [x.to(device) for x in batch]

            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds))

if __name__ == "__main__":
    evaluate(batch_size=32)  # adjust batch size if needed
