import os
import torch
import torch.nn.functional as F
import joblib
import re

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
# PATH
# ==============================
BASE_PATH = os.path.join(os.path.dirname(__file__), "models")

# ==============================
# LOAD MODELS
# ==============================
model_spam = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "spam_bert"),
    local_files_only=True
).to(device)
model_spam.eval()

model_category = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "category_bert"),
    local_files_only=True
).to(device)
model_category.eval()

model_priority = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BASE_PATH, "priority_bert"),
    local_files_only=True
).to(device)
model_priority.eval()

# ==============================
# TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(BASE_PATH, "tokenizer"),
    local_files_only=True
)

# ==============================
# BART SUMMARIZER
# ==============================
bart_tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(BASE_PATH, "bart_tokenizer"),
    local_files_only=True
)

bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    os.path.join(BASE_PATH, "bart_summarizer"),
    local_files_only=True
).to(device)
bart_model.eval()

# ==============================
# KEYBERT
# ==============================
kw_model = KeyBERT()

# ==============================
# TFIDF
# ==============================
tfidf = joblib.load(os.path.join(BASE_PATH, "tfidf.pkl"))

# ==============================
# CATEGORY MAP
# Must match training labels exactly:
#   0 = Work
#   1 = Meeting
#   2 = Finance
#   3 = Promotion
#   4 = Casual
# ==============================
CATEGORY_MAP = {
    0: "Work",
    1: "Meeting",
    2: "Finance",
    3: "Promotion",
    4: "Casual"
}

PRIORITY_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ==============================
# PREDICT FUNCTION
# ==============================
def predict(model, text):
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = torch.max(probs).item()

    return int(pred), float(conf)

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

    except Exception:
        return text[:120]

# ==============================
# KEYWORDS
# ==============================
def extract_keywords(text):
    try:
        clean_text = re.sub(r"[^a-zA-Z0-9₹ ]", " ", text)

        keywords = kw_model.extract_keywords(
            clean_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=7,
            use_mmr=True,
            diversity=0.7
        )

        phrases = [k[0] for k in keywords]

        txn_ids = re.findall(r"TXN\d+", text)
        amounts = re.findall(r"₹[\d,]+", text)

        return list(dict.fromkeys(phrases + txn_ids + amounts))

    except Exception:
        return []

# ==============================
# IMPORTANCE SCORE
# ==============================
def importance_score(text):
    text = text.lower()
    score = 0

    if "urgent" in text:
        score += 40
    if "deadline" in text:
        score += 30
    if "immediately" in text:
        score += 20
    if "meeting" in text:
        score += 20

    return min(score, 100)

# ==============================
# SPAM DETECTION (RULE-BASED)
# Uses specific multi-word phrases only.
# Bare words like "free" and "win" are removed
# to avoid false positives on casual emails like
# "Are you free this Saturday?"
# ==============================
def rule_based_spam(text):
    lower = text.lower()

    spam_phrases = [
        # Phishing / account threats
        "verify pan",
        "verify your aadhaar",
        "verify your kyc",
        "verify your identity",
        "kyc update",
        "kyc verification",
        "account blocked",
        "account will be blocked",
        "account suspended",
        "account suspension",
        "password expired",
        "reset your password",
        "secure link",
        "click the secure",
        "otp verification",
        "enter your otp",

        # Prize / reward scams
        "you have won",
        "you've won",
        "you are selected",
        "you have been selected",
        "claim your prize",
        "claim your reward",
        "claim your cashback",
        "claim now",
        "cash reward",
        "win a prize",
        "win now",
        "free gift",
        "free iphone",
        "free voucher",
        "free reward",
        "free cashback",
        "limited offer",
        "limited time offer",
        "congratulations you won",
        "congratulations you have been selected",

        # Financial scams
        "earn money online",
        "make money from home",
        "no experience needed",
        "work from home earn",
        "investment guaranteed",
        "double your money",

        # Generic spam signals
        "buy now and get",
        "click here to claim",
        "visit www",
        "click the link below",
        "act now",
        "hurry now",
        "today only",
        "offer expires tonight",
        "expires tonight"
    ]

    return any(phrase in lower for phrase in spam_phrases)


# ==============================
# RULE-BASED CATEGORY OVERRIDE
# Order matters — checked top to bottom:
#   1. Meeting  (most specific signals)
#   2. Promotion (before Finance — ₹ appears in
#                 promo emails too, so Finance
#                 must not match first)
#   3. Finance
#   4. Work
#   5. None → falls back to model
#
# Bare "free" and "win" removed from Promotion
# to avoid casual emails ("Are you free?") being
# wrongly labelled as Promotion.
# ==============================
def rule_based_category(text):
    lower = text.lower()

    # ----- MEETING -----
    if any(w in lower for w in [
        "meeting", "schedule", "discussion",
        "appointment", "agenda", "invite",
        "google meet", "zoom", "conference call"
    ]):
        return 1  # Meeting

    # ----- PROMOTION -----
    # No bare "free" or "win" here
    promo_words = [
        "offer", "discount", "sale", "promotion",
        "cashback", "reward", "voucher", "coupon",
        "deal", "limited time", "congratulations",
        "prize", "claim", "mega sale", "shop now",
        "% off", "free gift", "free voucher",
        "free iphone", "free reward",
        "win a prize", "win now", "you have won",
        "buy now", "hurry", "expires tonight"
    ]
    if any(w in lower for w in promo_words):
        return 3  # Promotion

    # ----- FINANCE -----
    # ₹ intentionally removed — promo emails use it too
    finance_words = [
        "invoice", "payment", "bank", "finance",
        "transaction", "upi", "neft", "imps",
        "debit", "credit", "balance", "salary",
        "refund", "amount debited", "amount credited",
        "account statement", "bill"
    ]
    if any(w in lower for w in finance_words):
        return 2  # Finance

    # ----- WORK -----
    work_words = [
        "project", "team", "update", "report",
        "deadline", "task", "sprint", "jira",
        "ticket", "deployment", "server", "backend",
        "api", "bug", "fix", "release", "production",
        "client", "manager", "submission", "document"
    ]
    if any(w in lower for w in work_words):
        return 0  # Work

    return None  # No rule matched → let model decide


# ==============================
# HOME
# ==============================
@app.route("/")
def home():
    return "AI Email Analyzer Running ✅"


# ==============================
# ANALYZE EMAIL
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze_email():

    data = request.json
    text = data.get("email", "")

    if not text:
        return jsonify({"error": "No email provided"}), 400

    lower_text = text.lower()

    # ======================
    # SPAM
    # Rule-based check first.
    # If rules say spam → override model.
    # If model confidence < 0.85 → not spam
    # (avoids false positives on casual emails).
    # ======================
    spam_pred, spam_conf = predict(model_spam, text)
    spam = bool(spam_pred)

    if rule_based_spam(text):
        spam = True
        spam_conf = max(spam_conf, 0.95)
    elif spam_conf < 0.85:
        spam = False

    # ======================
    # CATEGORY
    # Rule-based first to fix model bias toward Work.
    # Falls back to model if no rule matches.
    # ======================
    rule_cat = rule_based_category(text)

    if rule_cat is not None:
        cat_pred = rule_cat
        cat_conf = 0.95
    else:
        cat_pred, cat_conf = predict(model_category, text)
        if isinstance(cat_pred, str) and cat_pred.upper().startswith("LABEL_"):
            cat_pred = int(cat_pred.split("_")[1])
        cat_pred = int(cat_pred)

    category = CATEGORY_MAP.get(cat_pred, "Casual")

    # ======================
    # PRIORITY
    # ======================
    prio_pred, prio_conf = predict(model_priority, text)
    prio_pred = int(prio_pred)
    priority = PRIORITY_MAP.get(prio_pred, "Low")

    urgent_words = [
        "immediately", "today only", "before midnight",
        "expires tonight", "2 hours", "30 minutes",
        "urgent", "asap", "critical", "emergency",
        "server down", "action required", "response needed"
    ]

    if any(w in lower_text for w in urgent_words):
        priority = "High"
        prio_conf = max(prio_conf, 0.95)

    # ======================
    # SUMMARY
    # ======================
    summary = summarize_email(text)

    # ======================
    # KEYWORDS
    # ======================
    keywords = extract_keywords(text)

    # ======================
    # IMPORTANCE
    # ======================
    importance = importance_score(text)
    if priority == "High":
        importance = max(importance, 80)

    # ======================
    # RESPONSE
    # ======================
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
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)