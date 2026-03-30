import os
import torch
import torch.nn.functional as F
import joblib

from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from keybert import KeyBERT

# ==============================
# INIT APP
# ==============================
app = Flask(__name__)

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# PATH SETUP
# ==============================
BASE_PATH = os.path.join(os.path.dirname(__file__), "models")

# ==============================
# LOAD MODELS
# ==============================
model_spam = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "spam_bert"),
    local_files_only=True
).to(device)

model_category = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "category_bert"),
    local_files_only=True
).to(device)

model_priority = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "priority_bert"),
    local_files_only=True
).to(device)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(BASE_PATH, "tokenizer"),
    local_files_only=True
)

# BART
bart_tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(BASE_PATH, "bart_tokenizer"),
    local_files_only=True
)

bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    os.path.join(BASE_PATH, "bart_summarizer"),
    local_files_only=True
).to(device)

# KEYBERT
kw_model = KeyBERT()

# TFIDF
tfidf = joblib.load(os.path.join(BASE_PATH, "tfidf.pkl"))

# EVAL MODE
model_spam.eval()
model_category.eval()
model_priority.eval()

# ==============================
# LABEL MAPS
# ==============================
category_map = {
    0: "Work",
    1: "Meeting",
    2: "Finance",
    3: "Promotion",
    4: "Casual"
}

priority_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ==============================
# PREDICT FUNCTION
# ==============================
def predict(model, text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        conf = torch.max(probs).item()

    return pred, conf

# ==============================
# SUMMARY
# ==============================
def summarize_email(text):
    try:
        inputs = bart_tokenizer(
            [text[:1000]],
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        summary_ids = bart_model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=60,
            min_length=15
        )

        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except:
        return text[:120]

# ==============================
# KEYWORDS
# ==============================
def extract_keywords(text):
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )
        return [kw[0] for kw in keywords]
    except:
        return []

# ==============================
# IMPORTANCE
# ==============================
def importance_score(text):
    text = text.lower()
    score = 0

    if "urgent" in text:
        score += 40
    if "deadline" in text:
        score += 30
    if "meeting" in text:
        score += 20
    if "immediately" in text:
        score += 20

    return min(score, 100)

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return "Flask backend running ✅"

# ==============================
# ANALYZE ROUTE
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze_email():

    data = request.json
    text = data.get("email", "")

    if not text:
        return jsonify({"error": "No email provided"}), 400

    lower_text = text.lower()

    # SPAM
    spam_pred, spam_conf = predict(model_spam, text)
    spam = bool(spam_pred)

    spam_override_words = [
        "verify pan", "aadhaar", "kyc", "otp",
        "account blocked", "password expired",
        "reset here", "secure link",
        "double money", "crypto investment",
        "limited offer", "70% off", "coupon",
        "claim prize", "winner", "cash reward"
    ]

    if any(word in lower_text for word in spam_override_words):
        spam = True
        spam_conf = max(spam_conf, 0.95)

    # CATEGORY
    cat_pred, cat_conf = predict(model_category, text)
    category = category_map.get(cat_pred, "Casual")

    # PRIORITY
    prio_pred, prio_conf = predict(model_priority, text)
    priority = priority_map.get(prio_pred, "Low")

    urgent_words = [
        "immediately", "30 minutes", "2 hours",
        "today only", "before midnight",
        "limited offer", "expires tonight"
    ]

    if any(word in lower_text for word in urgent_words):
        priority = "High"
        prio_conf = max(prio_conf, 0.95)

    # SUMMARY
    summary = summarize_email(text)

    # KEYWORDS
    keywords = extract_keywords(text)

    # IMPORTANCE
    importance = importance_score(text)
    if priority == "High":
        importance = max(importance, 80)

    return jsonify({
        "spam": spam,
        "spam_confidence": round(spam_conf, 2),
        "category": category,
        "category_confidence": round(cat_conf, 2),
        "priority": priority,
        "priority_confidence": round(prio_conf, 2),
        "keywords": keywords,
        "importance_score": importance,
        "summary": summary
    })

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)