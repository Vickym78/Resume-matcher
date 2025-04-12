import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import io

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

# Streamlit UI
st.title("🧠 Resume Matcher")
st.markdown("Match your resume with a job description using semantic similarity.")

# Upload resume
uploaded_resume = st.file_uploader("Upload your resume (PDF only)", type="pdf")

# Text input for job description
job_desc = st.text_area("Paste the Job Description here")

if uploaded_resume and job_desc:
    # Extract resume text
    resume_text = extract_text_from_pdf(uploaded_resume)

    # Encode both texts
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    similarity_percent = round(similarity_score * 100, 2)

    st.metric("Matching Score (%)", f"{similarity_percent}%")

    if similarity_percent > 75:
        st.success("Great match! 🎯")
    elif similarity_percent > 50:
        st.info("Decent match. Can be improved.")
    else:
        st.warning("Low match. Try tweaking your resume.")
else:
    st.info("Please upload a resume and paste the job description to continue.")
