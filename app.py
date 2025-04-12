import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Title
st.title("📄 Resume - Job Matcher")

# Sidebar
st.sidebar.title("Upload Resume PDFs")
uploaded_files = st.sidebar.file_uploader("Upload resume PDFs", type=["pdf"], accept_multiple_files=True)

# Input job description
st.subheader("📌 Paste the Job Description")
job_description = st.text_area("Enter job description here:")

# Handle resume files
if uploaded_files:
    resumes = []
    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        resumes.append({"name": uploaded_file.name, "text": resume_text})

    # Match button
    if st.button("🔍 Find Top Matches"):
        if job_description.strip() == "":
            st.warning("Please enter a job description.")
        else:
            # Encode JD
            jd_embedding = model.encode(job_description, convert_to_tensor=True)

            # Calculate similarity scores
            results = []
            for resume in resumes:
                res_embedding = model.encode(resume["text"], convert_to_tensor=True)
                score = util.pytorch_cos_sim(jd_embedding, res_embedding).item()
                results.append((resume["name"], resume["text"], round(score, 2)))

            # Sort by score
            results.sort(key=lambda x: x[2], reverse=True)

            # Display results
            st.subheader("🎯 Top Resume Matches")
            for i, (name, resume, score) in enumerate(results):
                st.markdown(f"### Rank #{i+1}: {name}")
                st.markdown(f"**Match Score:** `{score}`")
                st.markdown(f"**Resume Summary (First 500 characters):** {resume[:500]}...")
                st.markdown("---")
else:
    st.info("Please upload resume PDFs.")
