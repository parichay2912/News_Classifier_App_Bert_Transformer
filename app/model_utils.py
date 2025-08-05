# app/model_utils.py
from transformers import BertForSequenceClassification, BertTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("model/saved_model/")
model = BertForSequenceClassification.from_pretrained("model/saved_model/")
model.to(device)
model.eval()

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def predict_topic(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        predicted = torch.argmax(output.logits, dim=1).item()
    return label_map[predicted]
