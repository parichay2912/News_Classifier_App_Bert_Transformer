 News Classifier App using BERT Transformer

This project is a smart news classification web application that uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to classify news articles into categories like World, Sports, Business, and Science/Technology.
  
 Features

    Built using Hugging Face Transformers (BERTForSequenceClassification)

    Fine-tuned for multiclass news classification

    FastAPI backend for real-time predictions

    React frontend for dynamic user interaction

    Tokenizer and model loaded from a local pre-trained directory

    Unit tested using pytest

    Modular and extensible architecture

   Model

    Model: bert-base-uncased fine-tuned on news data

    Target Classes:

        0: World

        1: Sports

        2: Business

        3: Science/Technology

 Project Structure

smart_news_classifier/
├── app/
│   ├── main.py               # FastAPI app routes
│   └── model_utils.py        # Loads tokenizer & model
├── frontend/                 # React.js frontend
│   ├── src/
│   └── public/
├── model/
│   └── saved_model/          # Fine-tuned BERT model & config
├── tests/
│   └── test_prediction.py    # Unit tests using pytest
├── .gitignore
├── requirements.txt
└── README.md

 Frontend: React

    A clean, responsive React.js interface

    Allows users to:

        Enter custom article text

        View predicted category

        Display summary and headline optimization results (if enabled)

    Communicates with the FastAPI backend via HTTP requests (Axios/fetch)

 Backend: FastAPI

    Serves REST API endpoints:

        /predict – Classifies input text

        /summarize – (optional) Returns a summary using GPT-4o

        /optimize-headline – (optional) Suggests a refined title

    Loads and serves a locally saved fine-tuned BERT model

    Built for speed, modularity, and production readiness

🔧 Installation
1. Clone the repository

git clone https://github.com/parichay2912/News_Classifier_App_Bert_Transformer.git
cd News_Classifier_App_Bert_Transformer

2. Create a virtual environment

conda create -n news_classifier_env python=3.9
conda activate news_classifier_env

3. Install dependencies

pip install -r requirements.txt

🚦 Run the FastAPI App

uvicorn app.main:app --reload

    Open your browser at: http://127.0.0.1:8000/docs

    Use Swagger UI to test predictions.

Run Tests

pytest tests/

Sample Input

POST /predict

{
  "text": "NASA launches new space mission to Mars."
}

Response:

{
  "label": "Science/Technology"
}

 .gitignore

Excludes:

    Virtual environment files

    __pycache__/

    VSCode settings

    model/saved_model/

 License

This project is licensed under the MIT License.