import fitz
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


# ----------------------------
# Extract Text From PDF
# ----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ----------------------------
# Skill List
# ----------------------------
SKILL_LIST = [
    "python", "machine learning", "deep learning",
    "tensorflow", "pytorch", "docker", "kubernetes",
    "aws", "azure", "sql", "nlp", "data science",
    "flask", "django", "pandas", "numpy", "scikit-learn"
]


# ----------------------------
# Extract Skills
# ----------------------------
def extract_skills(text):
    found_skills = []
    for skill in SKILL_LIST:
        if skill in text:
            found_skills.append(skill)
    return found_skills


# ----------------------------
# Extract Experience
# ----------------------------
def extract_experience(text):
    match = re.search(r'(\d+)\s+years', text)
    if match:
        return int(match.group(1))
    return 0


# ----------------------------
# Match Score (Semantic)
# ----------------------------
def calculate_match_score(resume_text, jd_text):
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)
    score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return round(score * 100, 2)


# ----------------------------
# Weighted Final Score
# ----------------------------
def weighted_score(match_score, experience_years):
    experience_weight = min(experience_years * 2, 20)
    final = match_score + experience_weight
    return min(final, 100)
