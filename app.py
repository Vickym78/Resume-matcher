import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Title
st.title("📄 Resume - Job Matcher")

# Sidebar
st.sidebar.title("Upload Resume Dataset")
uploaded_file = st.sidebar.file_uploader("Upload resumes.csv", type=["csv"])

# Load default or uploaded resumes
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Fallback to default
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana"],
        "Resume": [
            "Data analyst with Python, SQL, PowerBI and machine learning.",
            "Backend engineer with Java, Spring Boot, and REST APIs.",
            "NLP engineer skilled in BERT, transformers, and text classification.",
            "Data scientist with deep learning, PyTorch, and statistics experience."
        ]
    })

# Input job description
st.subheader("📌 Paste the Job Description")
job_description = st.text_area("Enter job description here:")

# Match button
if st.button("🔍 Find Top Matches"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Encode JD
        jd_embedding = model.encode(job_description, convert_to_tensor=True)

        results = []
        for _, row in df.iterrows():
            res_embedding = model.encode(row['Resume'], convert_to_tensor=True)
            score = util.pytorch_cos_sim(jd_embedding, res_embedding).item()
            results.append((row['Name'], row['Resume'], round(score, 2)))

        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)

        # Show results
        st.subheader("🎯 Top Resume Matches")
        for i, (name, resume, score) in enumerate(results):
            st.markdown(f"### Rank #{i+1}: {name}")
            st.markdown(f"**Match Score:** `{score}`")
            st.markdown(f"**Resume Summary:** {resume}")
            st.markdown("---")
