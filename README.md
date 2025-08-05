

# 📰 News Classifier App using BERT Transformer

This project is a **smart news classification web application** that uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to classify news articles into categories like **World, Sports, Business, and Science/Technology**.

## Features

- Built using **Hugging Face Transformers** (`BERTForSequenceClassification`)
- Fine-tuned for multiclass classification on news headlines/articles
- Interactive **FastAPI** backend for serving predictions
- Tokenizer and model loaded from a pre-trained local directory
- Easy testing using **pytest**
- Simple and extensible architecture

##  Model

- Model: `bert-base-uncased` fine-tuned for news classification
- Classes:
  - `0`: World
  - `1`: Sports
  - `2`: Business
  - `3`: Science/Technology

##  Project Structure

smart_news_classifier/
├── app/
│ ├── main.py # FastAPI app routes
│ └── model_utils.py # Loads tokenizer & model
├── model/
│ └── saved_model/ # Contains fine-tuned BERT model & config
├── tests/
│ └── test_prediction.py # Unit tests using pytest
├── .gitignore
├── requirements.txt
└── README.md


## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/parichay2912/News_Classifier_App_Bert_Transformer.git
cd News_Classifier_App_Bert_Transformer

2. Create a virtual environment

conda create -n news_classifier_env python=3.9
conda activate news_classifier_env

3. Install dependencies

pip install -r requirements.txt

🚦 Run the App

uvicorn app.main:app --reload

Open your browser at: http://127.0.0.1:8000/docs

Use the Swagger UI to test predictions.
🧪 Run Tests

pytest tests/

🧾 Sample Input

POST /predict

{
  "text": "NASA launches new space mission to Mars."
}

Response:

{
  "label": "Science/Technology"
}

 .gitignore

Includes:

    Virtual environment files

    __pycache__

    VSCode settings

    model/saved_model/

License

This project is licensed under the MIT License.
    Acknowledgments

    Hugging Face Transformers

    FastAPI

    PyTorch


--