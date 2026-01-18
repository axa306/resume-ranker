# Resume Ranking & Bias Audit Tool

This Streamlit web app allows users to upload resumes and a job description, rank candidates by similarity using semantic embeddings, and audit for potential biases by gender and school type.

## Features

- Upload multiple resumes in PDF or CSV format
- Upload a plain text job description
- Extracts candidate name, gender, and school type from resumes
- Ranks candidates based on similarity to job description using Sentence Transformers (`MiniLM`)
- Bias auditing by:
  - Gender
  - School type (elite vs non-elite)
- Interactive filtering, resume preview, and scoring interpretation

## How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/axa306/resume-ranker-app.git
cd resume-ranker-app
