import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import re
from nameparser import HumanName
from gender_guesser.detector import Detector

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
detector = Detector(case_sensitive=False)

# Define elite school list
elite_schools = [
    "Harvard University", "Stanford University", "MIT", "Yale University",
    "Princeton University", "Columbia University", "University of Chicago",
    "California Institute of Technology", "University of Oxford", "University of Cambridge"
]

# Helper functions
def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines[:10]:  # Scan the first 10 lines for flexibility
        cleaned_line = re.sub(r'^(Name|NAME|name)\s*[:\-]*\s*', '', line).strip()
        possible_name = HumanName(cleaned_line)
        if possible_name.first and possible_name.last:
            return f"{possible_name.first} {possible_name.last}"
    return "Candidate 1"

def extract_school(text):
    for school in elite_schools:
        if school.lower() in text.lower():
            return school
    match = re.search(r'\b(?:University|College) of [A-Z][a-z]+', text)
    return match.group(0) if match else "unknown"

def extract_gender(name):
    first_name = name.split()[0]
    return detector.get_gender(first_name)

def extract_fields_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() or '' for page in pdf.pages)

    name = extract_name(text)
    school = extract_school(text)
    gender = extract_gender(name)
    school_type = "elite" if school in elite_schools else "non-elite" if school != "unknown" else "unknown"

    return {
        "name": name,
        "school": school,
        "school_type": school_type,
        "gender": gender,
        "text": text
    }

st.title("Resume Ranking & Bias Audit Tool")

# Sidebar: Upload job description and resumes
st.sidebar.header("Step 1: Upload Files")
job_desc_file = st.sidebar.file_uploader("Upload Job Description (.txt)", type="txt")
resume_files = st.sidebar.file_uploader("Upload Resumes (.csv or .pdf)", type=["csv", "pdf"], accept_multiple_files=True)

if job_desc_file and resume_files:
    job_desc = job_desc_file.read().decode("utf-8")
    resumes = []
    candidate_counter = 1

    for resume_file in resume_files:
        if resume_file.name.endswith(".csv"):
            df = pd.read_csv(resume_file)

            if "text" not in df.columns:
                if "Resume" in df.columns:
                    df["text"] = df["Resume"].fillna("")
                else:
                    df["text"] = ""

            if "name" not in df.columns:
                df["name"] = [f"Candidate {i+1}" for i in range(len(df))]

            for col in ["school", "school_type", "gender"]:
                if col not in df.columns:
                    df[col] = "unknown"

            resumes.append(df)

        elif resume_file.name.endswith(".pdf"):
            extracted = extract_fields_from_pdf(resume_file)
            resumes.append(pd.DataFrame([extracted]))

    resumes_df = pd.concat(resumes, ignore_index=True)

    # Encode job and resumes
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes_df["text"].tolist(), convert_to_tensor=True)

    similarities = util.cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()
    resumes_df["similarity_score"] = similarities
    ranked_df = resumes_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    # Bias Auditing
    st.header("Bias Auditing")
    if "gender" in ranked_df.columns and "school_type" in ranked_df.columns:
        gender_bias = ranked_df.groupby("gender")["similarity_score"].mean().reset_index()
        school_bias = ranked_df.groupby("school_type")["similarity_score"].mean().reset_index()

        st.subheader("Average Score by Gender")
        fig, ax = plt.subplots()
        ax.bar(gender_bias["gender"], gender_bias["similarity_score"])
        st.pyplot(fig)

        st.subheader("Average Score by School Type")
        fig, ax = plt.subplots()
        ax.bar(school_bias["school_type"], school_bias["similarity_score"])
        st.pyplot(fig)

    # Show Top Candidates
    st.header("Top Ranked Candidates")
    max_candidates = len(ranked_df)
    if max_candidates > 1:
        top_n = st.slider("Select number of candidates to display", 1, max_candidates, min(5, max_candidates))
    else:
        top_n = 1
        st.markdown("Only one candidate available.")

    st.dataframe(ranked_df[["name", "gender", "school", "school_type", "similarity_score"]].head(top_n))

    # Interpretation of Score Ranges
    st.subheader("Score Interpretation")
    st.markdown("""
    - **0.65 and above** → Very strong match  
    - **0.55 - 0.64** → Good match  
    - **0.45 - 0.54** → Moderate match  
    - **Below 0.45** → Weak or unrelated  
    """)

    # View Full Resume
    st.header("View Resume Details")
    selected_index = st.selectbox("Select a candidate to view full resume", range(top_n))
    selected_resume = ranked_df.iloc[selected_index]
    st.markdown(f"""
    ### 🔝 Top Resume Details  
    **Name:** {selected_resume["name"]}  
    **School:** {selected_resume["school"]}  
    **School Type:** {selected_resume["school_type"]}  
    **Similarity Score:** {selected_resume["similarity_score"]:.3f}  

    **📄 Resume Preview:**
    ```
    {selected_resume["text"][:2000]}
    ```
    """)
