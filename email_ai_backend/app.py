from flask import Flask, request, jsonify
import pandas as pd
import joblib
from transformers import pipeline
from keybert import KeyBERT

# ---------------------
# Initialize app
# ---------------------
app = Flask(__name__)

# ---------------------
# Load dataset
# ---------------------
df = pd.read_csv("final_email_ai_dataset.csv")

# ---------------------
# Load ML models
# ---------------------
spam_model = joblib.load("models/spam_model.pkl")
classifier = joblib.load("models/classifier.pkl")
priority_model = joblib.load("models/priority_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------------
# Keyword extraction model
# ---------------------
kw_model = KeyBERT()

# ---------------------
# Sentiment model
# ---------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="models/sentiment_model",
    tokenizer="models/sentiment_model"
)

# ---------------------
# Summarization model
# ---------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# ---------------------
# Helper functions
# ---------------------

def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = int(spam_model.predict(vec)[0])
    return "Yes" if pred == 1 else "No"


def predict_category(text):
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]

def predict_priority(text):

    text_lower = text.lower()

    high_words = [
        "urgent","asap","immediately","deadline",
        "before","today","tomorrow","important","action required"
    ]

    medium_words = [
        "meeting","schedule","call","discussion","review","update"
    ]

    if any(word in text_lower for word in high_words):
        return "High"

    if any(word in text_lower for word in medium_words):
        return "Medium"

    vec = vectorizer.transform([text])
    pred = int(priority_model.predict(vec)[0])

    if pred == 0:
        return "Low"
    elif pred == 1:
        return "Medium"
    else:
        return "High"

def predict_sentiment(text):

    text_lower = text.lower()

    # Rule-based overrides for email tone
    if "thank" in text_lower or "great" in text_lower or "appreciate" in text_lower:
        return "Positive"

    if "issue" in text_lower or "problem" in text_lower or "error" in text_lower:
        return "Negative"

    result = sentiment_pipeline(text)[0]
    label = result["label"].lower()

    if "neg" in label:
        return "Negative"
    elif "neu" in label:
        return "Neutral"
    else:
        return "Positive"


def summarize_email(text):

    # If email is too short, return original
    if len(text.split()) < 30:
        return text

    try:

        # Limit long emails for model
        text = text[:1000]

        summary = summarizer(
            text,
            max_length=60,
            min_length=20,
            do_sample=False
        )

        return summary[0]["summary_text"]

    except Exception as e:
        return "Summary not available"


def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, top_n=5)
    return [k[0] for k in keywords]


# ---------------------
# API Routes
# ---------------------

@app.route("/")
def home():
    return "Email AI Backend Running"


@app.route("/emails")
def emails():
    return jsonify(df.head(10).to_dict(orient="records"))


@app.route("/analyze", methods=["POST"])
def analyze_email():

    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    result = {
        "spam": predict_spam(text),
        "category": predict_category(text),
        "priority": predict_priority(text),
        "sentiment": predict_sentiment(text),
        "summary": summarize_email(text),
        "keywords": extract_keywords(text)
    }

    return jsonify(result)


# ---------------------
# Run server
# ---------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)