from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict_topic,improve_headline,generate_summary
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class NewsItem(BaseModel):
    title: str
    description: str
    model_name: str = "bert-base-uncased"  # default model

@app.get("/")
def read_root():
    return {"message": "News Classifier API is up and running!"}

@app.get("/model-info")
def get_model_info():
    return {
        "available_models": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "albert-base-v2"
        ],
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "framework": "Transformers + PyTorch"
    }

@app.post("/predict")
def classify_article(item: NewsItem):
    full_text = item.title + " " + item.description
    
    prediction,confidence = predict_topic(full_text, item.model_name)
  
    return {"model_used": item.model_name, "predicted_class_label": prediction,"confidence":confidence}

@app.post("/generate-summary")
def summarize_article(item: NewsItem):
    summary = generate_summary(item.description)
    return {
        "summary": summary
    }

@app.post("/improve-headline")
def improve_title(item: NewsItem):
    improved = improve_headline(item.title, item.description)
    return {
        "improved_headline": improved
    }
