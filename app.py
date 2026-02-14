import streamlit as st
from utils import *
import matplotlib.pyplot as plt

# -------------------------------
# MODEL LOAD
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# SKILL DICTIONARY
# -------------------------------
SKILL_LIST = [
    "python", "java", "machine learning",
    "deep learning", "sql", "aws",
    "docker", "tensorflow", "pytorch",
    "pandas", "numpy", "react", "fastapi"
]

# -------------------------------
# FUNCTIONS
# -------------------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    return list(set([skill for skill in SKILL_LIST if skill in text]))

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+years?', text)
    return int(matches[0]) if matches else 0

def calculate_match_score(resume_text, jd_text):
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)
    score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return round(score * 100, 2)

def weighted_score(score, experience):
    bonus = min(experience * 2, 15)
    return round(min(score + bonus, 100), 2)

# -------------------------------
# UI
# -------------------------------

st.title("🚀 AI Resume Intelligence System")

uploaded_files = st.file_uploader(
    "Upload Resume(s)",
    type="pdf",
    accept_multiple_files=True
)

jd_input = st.text_area("Paste Job Description")

if uploaded_files and jd_input:

    jd_text = clean_text(jd_input)

    results = []

    for uploaded_file in uploaded_files:

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        resume_text = extract_text_from_pdf(uploaded_file.name)
        resume_text = clean_text(resume_text)

        score = calculate_match_score(resume_text, jd_text)
        experience = extract_experience(resume_text)
        final_score = weighted_score(score, experience)

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        missing = list(set(jd_skills) - set(resume_skills))

        results.append({
            "name": uploaded_file.name,
            "score": score,
            "final_score": final_score,
            "experience": experience,
            "missing": missing
        })

    # ---------------------------
    # SORTING
    # ---------------------------
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    st.subheader("🏆 Resume Ranking")

    for i, res in enumerate(results, 1):

        col1, col2 = st.columns([3,1])

        with col1:
            st.markdown(f"### {i}. {res['name']}")

        with col2:
           st.metric("Final Score", f"{res['final_score']:.2f}%")


        # Color Feedback
        if res['final_score'] >= 80:
            st.success("Strong Match 🚀")
        elif res['final_score'] >= 60:
            st.warning("Moderate Match ⚡")
        else:
            st.error("Low Match ❌")

        st.write(f"Experience Detected: {res['experience']} years")

        if res['missing']:
            st.write("Missing Skills:")
            for skill in res['missing']:
                st.write(f"❌ {skill}")
        else:
            st.success("No Major Skills Missing 🎉")

        st.divider()

    # ---------------------------
    # SCORE CHART
    # ---------------------------
    st.subheader("📊 Score Comparison")

    names = [r["name"] for r in results]
    scores = [r["final_score"] for r in results]

    fig, ax = plt.subplots()
    ax.barh(names, scores)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Final Score (%)")
    st.pyplot(fig)
