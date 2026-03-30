
import torch
import torch.nn.functional as F

def importance_score(text):
    text = text.lower()
    score = 0
    if "urgent" in text: score += 40
    if "deadline" in text: score += 30
    if "meeting" in text: score += 20
    return min(score, 100)

def predict_with_confidence(model, tokenizer, text, device):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = torch.max(probs).item()
    return pred, conf
