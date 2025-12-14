import streamlit as st
import PyPDF2
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
from nltk.corpus import stopwords

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🧠",
    layout="wide"
)

# ------------------ FUNCTIONS ------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)


def extract_skills(text):
    skills_db = [
        "python", "machine learning", "deep learning", "sql",
        "nlp", "data science", "excel", "power bi",
        "tensorflow", "pytorch", "statistics"
    ]
    found_skills = [skill for skill in skills_db if skill in text]
    return found_skills


def calculate_similarity(resumes, jd):
    docs = resumes + [jd]
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(docs)
    scores = cosine_similarity(vectors[:-1], vectors[-1])
    return scores.flatten()


# ------------------ UI ------------------
st.title("🧠 AI Resume Screening System")
st.write(
    "Upload resumes and paste a job description to automatically "
    "rank candidates using **NLP & Machine Learning**."
)

st.divider()

# ------------------ INPUT SECTION ------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "📄 Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:
    job_description = st.text_area(
        "📝 Paste Job Description",
        height=200
    )

st.divider()

# ------------------ PROCESSING ------------------
if st.button("🔍 Screen Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("⚠️ Please upload resumes and paste a job description.")
    else:
        resume_texts = []
        resume_names = []
        resume_skills = []

        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            cleaned = clean_text(raw_text)
            skills = extract_skills(cleaned)

            resume_texts.append(cleaned)
            resume_names.append(file.name)
            resume_skills.append(", ".join(skills) if skills else "Not found")

        jd_clean = clean_text(job_description)
        scores = calculate_similarity(resume_texts, jd_clean)

        results_df = pd.DataFrame({
            "Candidate": resume_names,
            "Match %": (scores * 100).round(2),
            "Extracted Skills": resume_skills
        }).sort_values(by="Match %", ascending=False)

        # ------------------ RESULTS ------------------
        st.subheader("📊 Resume Ranking Results")
        st.dataframe(results_df, use_container_width=True)

        # ------------------ CHARTS ------------------
        st.subheader("📈 Match Percentage Visualization")

        fig, ax = plt.subplots()
        ax.barh(results_df["Candidate"], results_df["Match %"])
        ax.set_xlabel("Match Percentage")
        ax.set_ylabel("Candidate")
        ax.invert_yaxis()

        st.pyplot(fig)

        # ------------------ TOP CANDIDATE ------------------
        top_candidate = results_df.iloc[0]
        st.success(
            f"🏆 **Top Match:** {top_candidate['Candidate']} "
            f"({top_candidate['Match %']}%)"
        )

st.divider()

# ------------------ FOOTER ------------------
st.caption(
    "🚀 Built using Python, NLP, TF-IDF & Streamlit | "
    "Perfect for AIML final-year projects & internships"
)
