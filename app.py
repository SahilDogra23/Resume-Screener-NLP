import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import joblib
import numpy as np

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screener")
st.markdown("Match your resume against job descriptions and find missing skills!")
st.divider()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_match_score(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([
        clean_text(resume_text),
        clean_text(job_text)
    ])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(score * 100, 2)

def get_missing_skills(resume_text, job_text):
    skills = ['python', 'sql', 'machine learning', 'deep learning',
              'tensorflow', 'pytorch', 'docker', 'kubernetes', 'aws',
              'tableau', 'powerbi', 'excel', 'nlp', 'statistics',
              'pandas', 'numpy', 'scikit', 'react', 'java', 'spark',
              'hadoop', 'mongodb', 'postgresql', 'git', 'agile']
    
    job_words = set(clean_text(job_text).split())
    resume_words = set(clean_text(resume_text).split())
    
    missing = [skill for skill in skills 
               if skill in job_words and skill not in resume_words]
    present = [skill for skill in skills 
               if skill in job_words and skill in resume_words]
    
    return missing, present

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Your Resume")
    resume_text = st.text_area(
        "Paste your resume here",
        height=300,
        placeholder="Paste your resume text here..."
    )

with col2:
    st.subheader("💼 Job Description")
    job_text = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="Paste the job description here..."
    )

st.divider()

if st.button("🔍 Analyze Match", use_container_width=True, type="primary"):
    if resume_text and job_text:
        # Match score
        score = get_match_score(resume_text, job_text)
        missing, present = get_missing_skills(resume_text, job_text)

        # Display score
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match Score", f"{score}%")
        with col2:
            st.metric("Skills Present", len(present))
        with col3:
            st.metric("Skills Missing", len(missing))

        st.divider()

        # Color code the score
        if score >= 50:
            st.success(f"✅ Strong match! Your resume matches {score}% of the job description.")
        elif score >= 25:
            st.warning(f"⚠️ Moderate match. Your resume matches {score}% of the job description.")
        else:
            st.error(f"❌ Weak match. Your resume matches only {score}% of the job description.")

        # Skills analysis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Matching Skills")
            if present:
                for skill in present:
                    st.success(f"✓ {skill}")
            else:
                st.info("No matching skills found")

        with col2:
            st.subheader("❌ Missing Skills")
            if missing:
                for skill in missing:
                    st.error(f"✗ {skill}")
            else:
                st.success("No missing skills — great match!")

        st.divider()
        st.caption("💡 Tip: Add missing skills to your resume if you have experience with them!")
    else:
        st.warning("Please paste both your resume and the job description!")