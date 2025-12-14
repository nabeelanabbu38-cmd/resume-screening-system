import streamlit as st
import PyPDF2
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords


# ---------- FUNCTIONS ----------

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


def calculate_similarity(resumes, job_desc):
    documents = resumes + [job_desc]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1], tfidf_matrix[-1]
    )
    return similarity_scores.flatten()


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="AI Resume Screening System", layout="centered")

st.title("🧠 AI Resume Screening System")
st.write("Upload resumes and paste a job description to get match percentage")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description Here")

if st.button("🔍 Screen Resumes"):
    if not uploaded_files or job_description.strip() == "":
        st.warning("Please upload resumes and enter a job description")
    else:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            cleaned = clean_text(text)
            resume_texts.append(cleaned)
            resume_names.append(file.name)

        cleaned_jd = clean_text(job_description)
        scores = calculate_similarity(resume_texts, cleaned_jd)

        results = sorted(
            zip(resume_names, scores),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("📊 Match Results")
        for name, score in results:
            st.write(f"📄 **{name}** → **{round(score * 100, 2)}% match**")
