import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    text = extract_text(pdf_file)
    return text

# Function to compute similarity between job description and resume
def get_similarity(job_desc, resume_text):
    # Create the tf-idf vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Vectorize the job description and resume
    vectors = vectorizer.fit_transform([job_desc, resume_text])
    
    # Compute the cosine similarity
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    return cosine_sim[0][0]

# Streamlit UI for the app
st.title("Resume-Job Description Matcher")

# Input job description as text
job_desc = st.text_area("Enter Job Description (Copy-Paste)", height=200)

# Upload resume (PDF file)
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if job_desc and resume_file is not None:
    # Extract text from the uploaded resume PDF file
    resume_text = extract_text_from_pdf(resume_file)
    
    # Display extracted resume text (optional)
    st.subheader("Resume Text")
    st.write(resume_text)

    # Get the similarity score
    similarity = get_similarity(job_desc, resume_text)

    # Display the similarity score
    st.subheader("Similarity Score")
    st.write(f"The similarity between the resume and job description is: {similarity * 100:.2f}%")

