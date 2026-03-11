import streamlit as st
import requests
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Email Intelligence System",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI Email Intelligence System")
st.write("Analyze emails using Machine Learning for spam detection, sentiment, summary, keywords, and priority.")

# -------------------------------
# Email Input Section
# -------------------------------
email_text = st.text_area(
    "Paste your email text here:",
    height=200
)

if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter some email text!")
    else:
        # -------------------------------
        # Call Backend API
        # -------------------------------
        url = "http://127.0.0.1:5000/analyze"  # Your Flask API endpoint
        data = {"text": email_text}

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise error if status not 200
            result = response.json()

            # -------------------------------
            # Display Analysis Results
            # -------------------------------
            st.subheader("📊 Analysis Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Spam", result.get("spam", "N/A"))
            col2.metric("Category", result.get("category", "N/A"))
            col3.metric("Priority", result.get("priority", "N/A"))

            st.write("### Sentiment")
            st.info(result.get("sentiment", "N/A"))

            st.write("### Email Summary")
            st.success(result.get("summary", "N/A"))

            st.write("### Keywords")
            keywords = result.get("keywords", [])
            if keywords:
                for k in keywords:
                    st.write("•", k)
            else:
                st.write("No keywords extracted.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
        except ValueError:
            st.error("Invalid response from backend.")

# -------------------------------
# Dataset Preview Section
# -------------------------------
st.write("---")
st.write("### 📄 Dataset Preview")

try:
    data = pd.read_csv("final_email_ai_dataset.csv")
    st.dataframe(data.head(50))
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'final_email_ai_dataset.csv' is in the project folder.")