from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
import re 
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_cache = {}
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_model(model_name):
    if model_name not in models_cache:
        model_path = os.path.join("model", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        models_cache[model_name] = (tokenizer, model)
    return models_cache[model_name]

def predict_topic(text, model_name="bert-base-uncased"):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    class_labels = ["World", "Sports", "Business", "Sci/Tech"]
    pred_index = prediction
    predicted_label = class_labels[pred_index]
    
    return {predicted_label}



def generate_summary(description: str) -> str:
    prompt = (
        f"Generate the summary of description\n\n"
        f"Description: {description}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=60
    )
    output = response.choices[0].message.content.strip()
    return output.strip('"').strip()

def improve_headline(title: str, description: str) -> str:
    prompt = (
        f"Improve the following news headline based on the article description:\n\n"
        f"Title: {title}\n"
        f"Description: {description}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=20
    )
    output = response.choices[0].message.content.strip()
    return output.strip('"').strip()