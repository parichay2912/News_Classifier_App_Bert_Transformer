# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict_topic

app = FastAPI()

class NewsItem(BaseModel):
    title: str
    description: str

@app.get("/")
def read_root():
    return {"message": "News Classifier API is up and running!"}

@app.get("/model-info")
def get_model_info():
    return {
        "model_name": "bert-base-uncased",
        "version": "1.0",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "framework": "Transformers + PyTorch"
    }


@app.post("/predict")
def classify_article(item: NewsItem):
    prediction = predict_topic(item.title + " " + item.description)
    return {"predicted_class": prediction}
