import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System", layout="wide")

NICE_SKILLS = ["Streamlit", "TensorFlow", "Git", "Seaborn", "Deep Learning"]
SPECIAL = {"c++": "cplusplus", "c#": "csharp", ".net": "dotnet"}


# ---------------- Setup ----------------
@st.cache_resource
def load_nlp():
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_data
def load_defaults():
    with open("job_description.txt", encoding="utf-8") as f:
        job = f.read()
    with open("required_skills.txt", encoding="utf-8") as f:
        req = f.read()
    resumes = []
    for path in sorted(glob.glob("resumes/*.txt")):
        with open(path, encoding="utf-8") as f:
            resumes.append((os.path.basename(path), f.read()))
    return job, req, resumes


# ---------------- Text processing (same logic as the notebook) ----------------
def skill_token(skill):
    t = skill.lower().strip()
    for k, v in SPECIAL.items():
        t = t.replace(k, v)
    t = re.sub(r"[ \-]+", "_", t)
    return re.sub(r"[^a-z_]", "", t)


def build_maps(required):
    all_skills = required + NICE_SKILLS
    phrases = {}
    for s in all_skills:
        low = s.lower().strip()
        if (" " in low or "-" in low) and not any(k in low for k in SPECIAL):
            phrases[low] = skill_token(s)
    keep = {skill_token(s) for s in all_skills} | set(SPECIAL.values())
    return phrases, keep


def preprocess(text, phrases, keep, stop_words, lemmatizer, name=None):
    if name:
        text = re.sub(re.escape(name), " ", text, flags=re.I)  
    text = re.sub(r"\S+@\S+", " ", text)                        
    text = text.lower()
    for k, v in SPECIAL.items():
        text = text.replace(k, v)
    for k in sorted(phrases, key=len, reverse=True):
        text = text.replace(k, phrases[k])
    text = re.sub(r"[^a-z_\s]", " ", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    out = []
    for t in tokens:
        if t in keep:
            out.append(t)
        else:
            out.append(lemmatizer.lemmatize(lemmatizer.lemmatize(t), pos="v"))
    return " ".join(out)


def guess_name(text, fallback):
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first and len(first.split()) <= 4 and not re.search(r"[@\d:]", first):
        return first
    return fallback


def read_upload(f):
    if f.name.lower().endswith(".pdf"):
        reader = PdfReader(f)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return f.read().decode("utf-8", errors="ignore")


# ---------------- Page ----------------
default_job, default_req, sample_resumes = load_defaults()
stop_words, lemmatizer = load_nlp()

st.title(" AI Resume Screening System")
st.write("Rank resumes by how well they match a job description, using NLP "
         "(TF-IDF and cosine similarity) plus required-skill matching.")

st.sidebar.header("Settings")
source = st.sidebar.radio("Resumes", ["Use sample resumes", "Upload my own"])
w_cos = st.sidebar.slider("Weight of cosine similarity (%)", 0, 100, 50, 5) / 100
top_n = st.sidebar.slider("Shortlist size (top N)", 1, 30, 10)
auto_thr = st.sidebar.checkbox("Auto threshold (largest score gap)", value=True)
manual_thr = st.sidebar.slider("Manual threshold (match %)", 0, 100, 30,
                               disabled=auto_thr)
st.sidebar.caption("The rest of the weight goes to required-skill coverage.")

tab1, tab2 = st.tabs(["Screening", "About"])

with tab2:
    st.markdown("""
**How it works**
1. Each resume and the job description are cleaned: lowercase, punctuation removed,
   stopwords removed, words reduced to their base form. Skills like *machine learning*
   are kept as single tokens. Names and emails are removed before scoring.
2. TF-IDF turns every text into numbers, giving rare, meaningful words more weight.
3. **Cosine similarity** measures how similar each resume is to the job description.
4. **Keyword matching** checks how many required skills the resume contains.
5. Match score = weighted mix of the two, shown as a percentage. Candidates are ranked,
   then shortlisted by top N and a minimum score.

**Limitations**
- The match score is a **ranking aid, not a probability** of being a good hire.
- Keyword matching misses synonyms (for example "ML" or "sklearn").
- The sample resumes are synthetic and cleaner than real ones, so real-world accuracy
  would be lower.
- Use it to decide who to read first. A person should make the final decision.
- Don't upload real people's resumes to this public demo.
""")

with tab1:
    c1, c2 = st.columns([2, 1])
    job_text = c1.text_area("Job description", default_job, height=300)
    skills_text = c2.text_area("Required skills (one per line)", default_req, height=300)
    required = list(dict.fromkeys(s.strip() for s in skills_text.splitlines() if s.strip()))

    if not job_text.strip() or not required:
        st.warning("Enter a job description and at least one required skill.")
        st.stop()

    candidates = []
    if source == "Use sample resumes":
        candidates = list(sample_resumes)
        st.caption(f"Using {len(candidates)} synthetic sample resumes.")
    else:
        files = st.file_uploader("Upload resumes (TXT or PDF)", type=["txt", "pdf"],
                                 accept_multiple_files=True)
        for f in files or []:
            text = read_upload(f)
            if text.strip():
                candidates.append((f.name, text))
            else:
                st.warning(f"No text found in {f.name} (a scanned PDF?). Skipped.")

    if not candidates:
        st.info("No resumes to screen yet.")
        st.stop()

    # ---------- Clean ----------
    phrases, keep = build_maps(required)
    job_clean = preprocess(job_text, phrases, keep, stop_words, lemmatizer)

    rows = []
    for fname, text in candidates:
        name = guess_name(text, os.path.splitext(fname)[0])
        clean = preprocess(text, phrases, keep, stop_words, lemmatizer, name=name)
        rows.append({"Candidate": name, "Clean": clean})
    data = pd.DataFrame(rows)
    data["Row"] = range(len(data))

    # ---------- TF-IDF and cosine ----------
    docs = data["Clean"].tolist() + [job_clean]
    vec = TfidfVectorizer()
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        st.error("Not enough text to analyse. Check the job description and resumes.")
        st.stop()
    n = len(data)
    data["Cosine"] = cosine_similarity(matrix[:n], matrix[n]).ravel()

    # ---------- Keyword matching ----------
    req_tokens = {s: skill_token(s) for s in required}
    found_lists, missing_lists = [], []
    for clean in data["Clean"]:
        tokens = set(clean.split())
        found = [s for s, t in req_tokens.items() if t in tokens]
        found_lists.append(found)
        missing_lists.append([s for s in required if s not in found])
    data["Found"] = found_lists
    data["Missing"] = missing_lists
    data["Keyword"] = data["Found"].str.len() / len(required)

    # ---------- Score, rank, shortlist ----------
    data["Match"] = (w_cos * data["Cosine"] + (1 - w_cos) * data["Keyword"]) * 100
    data = data.sort_values("Match", ascending=False).reset_index(drop=True)
    data["Rank"] = data["Match"].rank(method="min", ascending=False).astype(int)

    scores = data["Match"].to_numpy()
    if auto_thr:
        if len(scores) >= 2:
            gaps = scores[:-1] - scores[1:]
            i = int(gaps.argmax())
            threshold = float((scores[i] + scores[i + 1]) / 2)
        else:
            threshold = 0.0
    else:
        threshold = float(manual_thr)

    def status(row):
        if row["Match"] < threshold:
            return "Filtered out"
        if row["Rank"] <= top_n:
            return "Shortlisted"
        return "Reserve"

    data["Status"] = data.apply(status, axis=1)

    # ---------- Output ----------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes screened", n)
    m2.metric("Shortlisted", int((data["Status"] == "Shortlisted").sum()))
    m3.metric("Reserve", int((data["Status"] == "Reserve").sum()))
    m4.metric("Threshold used", f"{threshold:.1f}%")

    table = pd.DataFrame({
        "Rank": data["Rank"],
        "Candidate": data["Candidate"],
        "Match %": data["Match"].round(1),
        "Cosine": data["Cosine"].round(3),
        "Skills found": data["Found"].str.len().astype(str) + f"/{len(required)}",
        "Status": data["Status"],
        "Missing skills": data["Missing"].apply(", ".join),
    })
    st.subheader("Ranking")
    st.dataframe(table)
    st.download_button("Download ranking as CSV", table.to_csv(index=False),
                       file_name="ranking.csv", mime="text/csv")

    colors = {"Shortlisted": "#2e8b57", "Reserve": "#f0ad4e", "Filtered out": "#b0b0b0"}
    plot_df = data.sort_values("Match")
    labels = plot_df["Rank"].astype(str) + ". " + plot_df["Candidate"]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(plot_df))))
    ax.barh(labels, plot_df["Match"], color=plot_df["Status"].map(colors))
    ax.axvline(threshold, color="red", linestyle="--")
    ax.set_xlabel("Match score (%)")
    ax.set_title(f"Candidate ranking (green = shortlisted, red line = threshold {threshold:.1f})")
    plt.tight_layout()
    st.pyplot(fig)

    # ---------- Inspect one candidate ----------
    st.subheader("Inspect a candidate")
    idx = st.selectbox("Candidate", list(data.index),
                       format_func=lambda i: f"#{data.loc[i, 'Rank']}  {data.loc[i, 'Candidate']}")
    row = data.loc[idx]
    d1, d2 = st.columns(2)
    d1.markdown("**Required skills found**")
    d1.write(", ".join(row["Found"]) or "none")
    d2.markdown("**Required skills missing**")
    d2.write(", ".join(row["Missing"]) or "none")

    contrib = matrix[int(row["Row"])].multiply(matrix[n]).toarray().ravel()
    names = vec.get_feature_names_out()
    top_idx = contrib.argsort()[::-1][:8]
    words = [names[j].replace("_", " ") for j in top_idx if contrib[j] > 0]
    st.caption("Words contributing most to the match: " + (", ".join(words) or "none"))
    st.caption("The match score is a ranking aid, not a probability of being hired.")