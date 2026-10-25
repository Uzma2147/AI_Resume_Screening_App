# 📄 AI Resume Screening System

An AI resume screening system that cleans resume text with NLP, converts it into TF-IDF vectors, and scores each candidate against a job description using cosine similarity and required-skill matching.
A Streamlit app ranks the candidates, shortlists the best matches, and shows which required skills each one is missing.


##  Problem Statement

Companies receive hundreds of resumes for a single job, and screening them by hand is slow. This project builds a system that reads resumes, compares each one with a job description, gives it a **match score (%)**, and produces a **ranked list and shortlist**, so a recruiter can read the most relevant resumes first.

##  Dataset

The data is "synthetic": one Data Scientist job description, a list of 10 required skills, and 30 generated resumes saved as TXT files. The resumes are split into three groups so the ranking can be checked:

| Group | Count | Description |
| Strong | 10 | Data scientists with most of the required skills |
| Medium | 10 | Analysts with some of the skills (Excel, Power BI, some SQL or Python) |
| Weak | 10 | Unrelated roles (design, accounting, teaching) with almost none of the skills |

The group label is stored separately and is **never used in scoring**. It is only used at the end to check whether the ranking makes sense. The groups overlap slightly on purpose, so the test is not too easy. All names and emails are invented.

##  Project Workflow

1. **Data collection:** generated the job description, required skills, and 30 resumes (TXT). Included a PDF text-extraction demo using `pypdf`.
2. **Text preprocessing (NLP):**
   - Lowercasing and removal of punctuation, digits, and special characters
   - Tokenization and stopword removal (NLTK)
   - Lemmatization (noun pass, then verb pass)
   - **Skill protection:** multi-word skills such as *machine learning* are kept as one token, special names such as `C++` and `C#` are preserved, and skill words are not lemmatized (so *pandas* does not become *panda*)
   - **Blind screening:** names and emails are removed before scoring
3. **Feature extraction:** Bag of Words and **TF-IDF** (kept), fitted on the resumes plus the job description.
4. **Matching algorithm:**
   - **Cosine similarity** between each resume and the job description
   - **Keyword matching** against the 10 required skills (whole-token matching)
   - Combined score: `Match % = (w × cosine + (1 − w) × keyword coverage) × 100`, with a default weight of 50/50
5. **Ranking and shortlisting:** ranked by match score. Candidates must pass both a top-N rule and a minimum score to be **Shortlisted**. Others go to **Reserve** or are **Filtered out**. The default threshold sits in the middle of the largest gap between neighboring scores.
6. **Validation:** checked the ranking against the hidden groups (Precision@10, pairwise ordering) and tested how sensitive the top candidates are to the score weights.
7. **Deployment:** Streamlit app on Streamlit Community Cloud.

##  Results


**Notes on reading the scores:**

- Cosine scores are relative. The job description is long, so even a strong resume overlaps with only part of it. The match score is for **ordering** candidates, not a probability of being hired.
- These results measure the pipeline on resumes I generated myself, with uniform wording. They do not show how accurate the system would be on real resumes.

##  The App

- **Editable job description and required skills**, so you can try a different role
- **Sample resumes or upload your own** (TXT or PDF)
- **Sidebar controls:** cosine-vs-keyword weight, shortlist size, and automatic or manual threshold
- **Ranking table** with match %, cosine score, skills found, status, and missing skills, plus a CSV download
- **Bar chart** of the ranking, colored by status, with the threshold line
- **Inspect a candidate:** found and missing skills, and the words that contributed most to the match

> Streamlit Community Cloud apps are public. Don't upload real people's resumes to the demo.

##  Technologies Used

Python, Pandas, NumPy, Scikit-learn (TF-IDF, cosine similarity), NLTK, pypdf, Matplotlib, Seaborn, Streamlit, Google Colab, VS Code, Git and GitHub


## What to Learned

- Text must be turned into numbers before it can be compared. TF-IDF gives rare, meaningful words more weight than common ones.
- Naive cleaning can damage skills: it splits "machine learning", turns `C++` into `c`, and lemmatizes "pandas" into "panda".
- Cosine similarity compares the mix of words, not the length of the document.
- Keyword matching is easy to explain but misses synonyms. Cosine similarity captures overall similarity but doesn't say what is missing. Combining the two is stronger, and the weights are a choice I should test, not a fixed truth.
- With no labels there is no accuracy score, so I validated the system with groups whose quality I already knew.
- Deployment has its own challenges, such as library versions, extra data downloads, and file paths.

##  Limitations and Responsible Use

- The resumes are synthetic and more uniform than real ones. Real resumes use different wording ("ML", "sklearn"), so real-world accuracy would be lower.
- Keyword matching does not understand synonyms, context, or how well a skill is actually used.
- Automated screening can be unfair if used blindly. Removing names and emails helps, but wording differences can still lower someone's score.
- Use this tool to decide **who to read first**. A person should make the final hiring decision.



---

*Made by Uzma Kainat*
