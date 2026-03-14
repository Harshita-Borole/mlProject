import streamlit as st
import requests
import pandas as pd

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="SentinMail AI",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------
# CSS STYLING
# ------------------------------------------------
st.markdown("""
<style>

/* Main background */

.stApp{
background-color:#0f172a;
color:white;
}

/* All text white */

h1,h2,h3,h4,p,label,span{
color:white !important;
}

/* Navigation bar */

.navbar{
display:flex;
justify-content:space-between;
align-items:center;
background:#1e3a8a;
padding:15px 30px;
border-radius:10px;
margin-bottom:25px;
}

.nav-title{
font-size:22px;
font-weight:700;
color:white;
}

/* Cards */

.card{
background:#1e293b;
padding:20px;
border-radius:10px;
border:1px solid #3b82f6;
margin-bottom:20px;
}

/* Text area */

textarea{
background:#1e293b !important;
color:white !important;
border:1px solid #3b82f6 !important;
}

/* Buttons */

.stButton>button{
background:#2563eb;
color:white;
border:none;
border-radius:8px;
height:40px;
font-weight:600;
}

.stButton>button:hover{
background:#1d4ed8;
}

/* Metrics */

[data-testid="stMetricLabel"]{
color:white !important;
}

[data-testid="stMetricValue"]{
color:#60a5fa !important;
font-weight:800;
}

/* Keyword tags */

.tag{
display:inline-block;
background:#2563eb;
color:white;
padding:6px 12px;
border-radius:20px;
margin:4px;
font-size:0.8rem;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TOP NAVIGATION BAR
# ------------------------------------------------
st.markdown("""
<div class="navbar">
<div class="nav-title">🛡️ SentinMail AI Email Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# PAGE SELECTOR (HORIZONTAL)
# ------------------------------------------------
page = st.radio(
    "",
    ["Email Analysis","Dataset"],
    horizontal=True
)

st.divider()

# ------------------------------------------------
# PAGE 1 EMAIL ANALYSIS
# ------------------------------------------------
if page == "Email Analysis":

    st.header("📧 Email Intelligence Dashboard")

    col1,col2 = st.columns(2)

    # INPUT
    with col1:

        st.subheader("Email Input")

        email_text = st.text_area(
            "Paste Email Content",
            height=300
        )

        analyze = st.button("Analyze Email")

    # OUTPUT
    with col2:

        st.subheader("Analysis Result")

        if analyze:

            if email_text.strip()=="":
                st.warning("Please enter email text")

            else:

                try:

                    url="http://127.0.0.1:5000/analyze"

                    with st.spinner("Running AI analysis..."):

                        response=requests.post(
                            url,
                            json={"text":email_text}
                        )

                        result=response.json()

                    # METRICS

                    m1,m2,m3 = st.columns(3)

                    m1.metric(
                        "Spam",
                        result.get("spam","N/A")
                    )

                    m2.metric(
                        "Category",
                        result.get("category","N/A")
                    )

                    m3.metric(
                        "Priority",
                        result.get("priority","N/A")
                    )

                    st.divider()

                    # SUMMARY

                    st.subheader("Email Summary")

                    st.markdown(
                        f"""
                        <div class="card">
                        {result.get("summary","No summary available")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # SENTIMENT

                    st.subheader("Sentiment")

                    st.success(
                        result.get("sentiment","Neutral")
                    )

                    st.divider()

                    # KEYWORDS

                    st.subheader("Keywords")

                    keywords=result.get("keywords",[])

                    if keywords:

                        tags=""

                        for k in keywords:
                            tags+=f'<span class="tag">{k}</span>'

                        st.markdown(tags,unsafe_allow_html=True)

                    else:
                        st.write("No keywords detected")

                except Exception as e:

                    st.error("Backend server not running")

# ------------------------------------------------
# PAGE 2 DATASET
# ------------------------------------------------
elif page == "Dataset":

    st.header("📊 Dataset Dashboard")

    try:

        df=pd.read_csv("final_email_ai_dataset.csv")

        s1,s2,s3=st.columns(3)

        s1.metric("Total Emails",len(df))

        if "category" in df.columns:
            s2.metric("Categories",df["category"].nunique())

        if "spam" in df.columns:
            spam_ratio=(df["spam"].sum()/len(df))*100
            s3.metric("Spam %",f"{round(spam_ratio,1)}%")

        st.divider()

        st.dataframe(
            df.head(1000),
            use_container_width=True
        )

        csv=df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Dataset",
            csv,
            "sentinmail_dataset.csv",
            "text/csv"
        )

    except:

        st.error("Dataset file not found")