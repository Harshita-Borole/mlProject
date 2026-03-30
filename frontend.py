import streamlit as st
import requests

# ==============================
# TITLE
# ==============================
st.set_page_config(page_title="Email AI Assistant", layout="wide")

st.title("📩 AI Email Analyzer")
st.write("Analyze emails for Spam, Priority, Category, Summary & Keywords")

# ==============================
# INPUT
# ==============================
email_text = st.text_area("✉️ Enter Email Text", height=200)

# ==============================
# BUTTON
# ==============================
if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter an email")
    else:
        try:
            # Call Flask API
            response = requests.post(
                "http://127.0.0.1:5000/analyze",
                json={"email": email_text}
            )

            result = response.json()

            # ==============================
            # OUTPUT
            # ==============================
            st.subheader("📊 Analysis Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🚨 Spam", result["spam"])
                st.metric("Confidence", result["spam_confidence"])

            with col2:
                st.metric("📂 Category", result["category"])
                st.metric("Confidence", result["category_confidence"])

            with col3:
                st.metric("⚡ Priority", result["priority"])
                st.metric("Confidence", result["priority_confidence"])

            st.write("---")

            st.subheader("🧠 Summary")
            st.info(result["summary"])

            st.subheader("🔑 Keywords")
            st.write(", ".join(result["keywords"]))

            st.subheader("🔥 Importance Score")
            st.progress(result["importance_score"] / 100)

        except Exception as e:
            st.error("Backend not running! Start Flask server first.")