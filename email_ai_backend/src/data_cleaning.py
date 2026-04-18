import pandas as pd
import os

# ------------------------------
# CREATE OUTPUT FOLDER
# ------------------------------
os.makedirs("email_ai_backend/data/processed_clean", exist_ok=True)

# ------------------------------
# ENRON
# ------------------------------
enron = pd.read_csv("email_ai_backend/data/processed/enron_clean_dataset.csv")

enron = enron.dropna()

enron.to_csv("email_ai_backend/data/processed_clean/enron_clean_dataset.csv", index=False)

# ------------------------------
# SPAM
# ------------------------------
spam = pd.read_csv("email_ai_backend/data/processed/spam_clean_dataset.csv")

spam = spam.dropna()

spam.to_csv("email_ai_backend/data/processed_clean/spam_clean_dataset.csv", index=False)

print("✅ Data cleaning completed")