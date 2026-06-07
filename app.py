"""
AI Resume Analyzer — Streamlit single-file app.

Run locally:
    cd Downloads
    python -m streamlit run "app (1).py"
"""
from __future__ import annotations
from PyPDF2 import PdfReader


import io
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# NLP setup (lazy + cached)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Skills database
# ---------------------------------------------------------------------------

SKILLS_DB: Dict[str, List[str]] = {
    # Programming languages
    "Python": ["python", "py"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Go": ["golang", "go"],
    "Rust": ["rust"],
    "SQL": ["sql"],
    "R": [" r ", "r programming"],
    # Web / frameworks
    "React": ["react", "react.js", "reactjs"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Express": ["express", "express.js"],
    # Data / ML
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data viz"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    # Cloud / DevOps
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration"],
    "Git": ["git", "github", "gitlab"],
    "Linux": ["linux", "unix"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    # Soft skills
    "Communication": ["communication", "communicator"],
    "Leadership": ["leadership", "leader", "led team"],
    "Teamwork": ["teamwork", "team player", "collaboration", "collaborative"],
    "Problem Solving": ["problem solving", "problem-solving"],
    "Critical Thinking": ["critical thinking"],
    "Time Management": ["time management"],
    "Adaptability": ["adaptability", "adaptable"],
    "Project Management": ["project management", "agile", "scrum", "kanban"],
}

ACTION_VERBS = {
    "developed", "implemented", "designed", "built", "created", "led", "managed",
    "optimized", "improved", "increased", "decreased", "reduced", "launched",
    "delivered", "architected", "engineered", "automated", "deployed", "migrated",
    "refactored", "scaled", "mentored", "spearheaded", "established", "drove",
}

CRITICAL_SKILL_HINTS = {
    "required", "must have", "must-have", "mandatory", "essential", "minimum",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def parse_resume(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    raise ValueError("Unsupported file format. Please upload PDF or DOCX.")


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    if not text:
        return []
    padded = f" {text.lower()} "
    found = set()
    for canonical, variants in SKILLS_DB.items():
        for v in variants:
            pattern = r"(?<![a-z0-9])" + re.escape(v.strip()) + r"(?![a-z0-9])"
            if re.search(pattern, padded):
                found.add(canonical)
                break
    return sorted(found)


def extract_critical_skills(jd_text: str, jd_skills: List[str]) -> List[str]:
    """Heuristic: skills mentioned near 'required'/'must have' are critical."""
    if not jd_text:
        return []
    lower = jd_text.lower()
    critical = set()
    sentences = re.split(r"[.\n;]", lower)
    for sent in sentences:
        if any(h in sent for h in CRITICAL_SKILL_HINTS):
            for skill in jd_skills:
                for v in SKILLS_DB[skill]:
                    if re.search(r"(?<![a-z0-9])" + re.escape(v.strip()) + r"(?![a-z0-9])", sent):
                        critical.add(skill)
                        break
    return sorted(critical) if critical else jd_skills[: min(5, len(jd_skills))]


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

EXP_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)", re.I),
    re.compile(r"(?:experience|exp)\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I),
]


def extract_years_of_experience(text: str) -> float:
    if not text:
        return 0.0
    found = []
    for pat in EXP_PATTERNS:
        for m in pat.finditer(text):
            try:
                found.append(float(m.group(1)))
            except ValueError:
                continue
        if found:
            break
    return max(found) if found else 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    skill_match: float = 0.0
    semantic: float = 0.0
    keyword: float = 0.0
    experience: float = 0.0
    quality: float = 0.0
    total: float = 0.0
    details: Dict[str, object] = field(default_factory=dict)


WEIGHTS = {
    "skill_match": 40,
    "semantic": 20,
    "keyword": 15,
    "experience": 15,
    "quality": 10,
}


def score_skill_match(resume_skills: List[str], jd_skills: List[str], critical: List[str]) -> Tuple[float, Dict]:
    if not jd_skills:
        return 0.0, {"matched": [], "missing": [], "critical_missing": []}
    matched = sorted(set(resume_skills) & set(jd_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))
    critical_missing = sorted(set(critical) - set(resume_skills))
    base = len(matched) / len(jd_skills)
    penalty = 0.1 * len(critical_missing)
    score = max(0.0, min(1.0, base - penalty)) * WEIGHTS["skill_match"]
    return score, {"matched": matched, "missing": missing, "critical_missing": critical_missing}


def score_semantic(resume_text: str, jd_text: str) -> Tuple[float, Dict]:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, {"similarity": 0.0}
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        m = vec.fit_transform([resume_text, jd_text])
        sim = float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except Exception:
        sim = 0.0
    return sim * WEIGHTS["semantic"], {"similarity": sim}


def score_keywords(resume_text: str, jd_text: str) -> Tuple[float, Dict]:
    if not resume_text or not jd_text:
        return 0.0, {"top_keywords": [], "coverage": 0.0}
    jd_tokens = [t for t in tokenize(jd_text) if t.isalpha() and len(t) > 2]
    resume_tokens = tokenize(resume_text)
    if not jd_tokens:
        return 0.0, {"top_keywords": [], "coverage": 0.0}
    freq = pd.Series(jd_tokens).value_counts().head(25)
    resume_set = set(resume_tokens)
    hits = sum(1 for kw in freq.index if kw in resume_set)
    coverage = hits / len(freq)
    return coverage * WEIGHTS["keyword"], {
        "top_keywords": list(freq.index),
        "matched_keywords": [k for k in freq.index if k in resume_set],
        "missing_keywords": [k for k in freq.index if k not in resume_set],
        "coverage": coverage,
    }


def score_experience(resume_years: float, jd_years: float) -> Tuple[float, Dict]:
    if jd_years <= 0:
        ratio = 1.0 if resume_years > 0 else 0.5
    else:
        ratio = min(1.0, resume_years / jd_years)
    return ratio * WEIGHTS["experience"], {"resume_years": resume_years, "required_years": jd_years}


QUANT_PATTERN = re.compile(r"(\d+%|\$\d[\d,]*|\d{2,})")


def score_quality(resume_text: str) -> Tuple[float, Dict]:
    if not resume_text:
        return 0.0, {}
    lower = resume_text.lower()
    has_projects = bool(re.search(r"\b(projects?|portfolio)\b", lower))
    verbs_used = sorted({v for v in ACTION_VERBS if re.search(rf"\b{v}\b", lower)})
    quant_hits = QUANT_PATTERN.findall(resume_text)
    has_education = bool(re.search(r"\b(education|bachelor|master|phd|b\.?sc|m\.?sc|degree)\b", lower))

    proj_pts = 1.0 if has_projects else 0.0
    verbs_pts = min(1.0, len(verbs_used) / 5)
    quant_pts = min(1.0, len(quant_hits) / 4)
    edu_pts = 1.0 if has_education else 0.0

    raw = (proj_pts * 0.30) + (verbs_pts * 0.30) + (quant_pts * 0.25) + (edu_pts * 0.15)
    return raw * WEIGHTS["quality"], {
        "has_projects": has_projects,
        "action_verbs": verbs_used,
        "quantified_count": len(quant_hits),
        "has_education": has_education,
    }


def compute_total_score(
    resume_text: str,
    jd_text: str,
    resume_skills: List[str],
    jd_skills: List[str],
    critical_skills: List[str],
    resume_years: float,
    jd_years: float,
) -> ScoreBreakdown:
    s_skill, d_skill = score_skill_match(resume_skills, jd_skills, critical_skills)
    s_sem, d_sem = score_semantic(resume_text, jd_text)
    s_kw, d_kw = score_keywords(resume_text, jd_text)
    s_exp, d_exp = score_experience(resume_years, jd_years)
    s_q, d_q = score_quality(resume_text)
    total = s_skill + s_sem + s_kw + s_exp + s_q
    return ScoreBreakdown(
        skill_match=s_skill,
        semantic=s_sem,
        keyword=s_kw,
        experience=s_exp,
        quality=s_q,
        total=total,
        details={"skill": d_skill, "semantic": d_sem, "keyword": d_kw, "experience": d_exp, "quality": d_q},
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def strength_label(total: float) -> str:
    if total >= 75:
        return "Strong"
    if total >= 50:
        return "Intermediate"
    return "Beginner"


def color_for(score: float, max_score: float) -> str:
    pct = score / max_score if max_score else 0
    if pct >= 0.75:
        return "🟢"
    if pct >= 0.5:
        return "🟡"
    return "🔴"


def build_feedback(b: ScoreBreakdown) -> List[str]:
    tips: List[str] = []
    skill = b.details["skill"]
    if skill["critical_missing"]:
        tips.append(f"❗ Add critical missing skills: {', '.join(skill['critical_missing'])}.")
    if skill["missing"]:
        tips.append(f"➕ Consider mentioning these JD skills if you have them: {', '.join(skill['missing'][:8])}.")
    if b.details["semantic"]["similarity"] < 0.3:
        tips.append("📝 Resume context differs from the JD. Mirror the JD's language and responsibilities.")
    kw = b.details["keyword"]
    if kw.get("missing_keywords"):
        tips.append(f"🔑 Weak keyword usage. Try integrating: {', '.join(kw['missing_keywords'][:8])}.")
    exp = b.details["experience"]
    if exp["required_years"] and exp["resume_years"] < exp["required_years"]:
        tips.append(f"📅 JD asks for ~{exp['required_years']:.0f} yrs; resume shows {exp['resume_years']:.0f}. Highlight relevant projects to compensate.")
    q = b.details["quality"]
    if not q.get("has_projects"):
        tips.append("📁 Add a dedicated Projects section with measurable outcomes.")
    if len(q.get("action_verbs", [])) < 5:
        tips.append("💪 Use more action verbs (developed, optimized, led, implemented…).")
    if q.get("quantified_count", 0) < 3:
        tips.append("📊 Quantify achievements with numbers, %, or $ impact.")
    if not q.get("has_education"):
        tips.append("🎓 Include an Education section.")
    if not tips:
        tips.append("✅ Strong alignment. Keep tailoring per role for top results.")
    return tips


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(b: ScoreBreakdown, resume_skills, jd_skills, feedback) -> str:
    lines = [
        "AI RESUME ANALYZER — DETAILED REPORT",
        "=" * 50,
        "",
        f"Overall ATS Score: {b.total:.1f} / 100",
        f"Strength Level:    {strength_label(b.total)}",
        "",
        "Score Breakdown",
        "-" * 50,
        f"Skill Match (40):       {b.skill_match:.1f}",
        f"Semantic Similarity(20):{b.semantic:.1f}",
        f"Keyword Optimization(15):{b.keyword:.1f}",
        f"Experience Match (15):  {b.experience:.1f}",
        f"Resume Quality (10):    {b.quality:.1f}",
        "",
        "Skills",
        "-" * 50,
        f"Resume Skills ({len(resume_skills)}): {', '.join(resume_skills) or '—'}",
        f"JD Skills ({len(jd_skills)}): {', '.join(jd_skills) or '—'}",
        f"Matched: {', '.join(b.details['skill']['matched']) or '—'}",
        f"Missing: {', '.join(b.details['skill']['missing']) or '—'}",
        f"Critical Missing: {', '.join(b.details['skill']['critical_missing']) or '—'}",
        "",
        "Experience",
        "-" * 50,
        f"Resume years detected: {b.details['experience']['resume_years']}",
        f"JD years required:     {b.details['experience']['required_years']}",
        "",
        "Recommendations",
        "-" * 50,
    ]
    lines.extend(f"- {t}" for t in feedback)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_score_chart(b: ScoreBreakdown):
    df = pd.DataFrame({
        "Component": ["Skill Match", "Semantic", "Keywords", "Experience", "Quality"],
        "Score": [b.skill_match, b.semantic, b.keyword, b.experience, b.quality],
        "Max": [WEIGHTS["skill_match"], WEIGHTS["semantic"], WEIGHTS["keyword"], WEIGHTS["experience"], WEIGHTS["quality"]],
    })
    df["Pct"] = df["Score"] / df["Max"]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", title="Points"),
            y=alt.Y("Component:N", sort="-x"),
            color=alt.Color(
                "Pct:Q",
                scale=alt.Scale(domain=[0, 0.5, 0.75, 1], range=["#ef4444", "#f59e0b", "#10b981", "#059669"]),
                legend=None,
            ),
            tooltip=["Component", "Score", "Max"],
        )
        .properties(height=220)
    )
    text = chart.mark_text(align="left", baseline="middle", dx=4).encode(text=alt.Text("Score:Q", format=".1f"))
    st.altair_chart(chart + text, use_container_width=True)


def render_skill_ratio(b: ScoreBreakdown, jd_skills: List[str]):
    matched = len(b.details["skill"]["matched"])
    missing = max(0, len(jd_skills) - matched)
    df = pd.DataFrame({"Status": ["Matched", "Missing"], "Count": [matched, missing]})
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=55)
        .encode(
            theta="Count:Q",
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Matched", "Missing"], range=["#10b981", "#ef4444"])),
            tooltip=["Status", "Count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

    st.title("🧠 AI Resume Analyzer")
    st.caption("Upload your resume, paste a job description, and get an ATS-style breakdown.")

    with st.sidebar:
        st.header("⚙️ Inputs")
        uploaded = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
        jd_text = st.text_area("Job Description", height=260, placeholder="Paste the job description here…")
        analyze = st.button("🚀 Analyze", type="primary", use_container_width=True)
        st.markdown("---")
        st.caption("Tip: a real JD with required skills & years gives the best signal.")

    if not analyze:
        st.info("Upload a resume and paste a job description, then click **Analyze**.")
        return

    if uploaded is None:
        st.error("Please upload a resume file (PDF or DOCX).")
        return
    if not jd_text or not jd_text.strip():
        st.error("Please paste the job description.")
        return

    try:
        with st.spinner("Parsing resume…"):
            resume_text = parse_resume(uploaded)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Failed to parse resume: {e}")
        return

    if not resume_text or len(resume_text.strip()) < 30:
        st.error("Couldn't extract enough text from the resume. Try another file.")
        return

    with st.spinner("Analyzing…"):
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(jd_text)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        critical = extract_critical_skills(jd_text, jd_skills)
        resume_years = extract_years_of_experience(resume_text)
        jd_years = extract_years_of_experience(jd_text)
        breakdown = compute_total_score(
            cleaned_resume, cleaned_jd, resume_skills, jd_skills, critical, resume_years, jd_years,
        )
        feedback = build_feedback(breakdown)

    # ---- Top metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATS Score", f"{breakdown.total:.1f} / 100")
    c2.metric("Strength", strength_label(breakdown.total))
    c3.metric("Skills Matched", f"{len(breakdown.details['skill']['matched'])}/{len(jd_skills) or 0}")
    c4.metric("Experience", f"{resume_years:.0f} yr" + (f" / {jd_years:.0f} req" if jd_years else ""))

    st.progress(min(1.0, breakdown.total / 100))

    st.markdown("---")

    # ---- Two-column dashboard ----
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("📊 Score Breakdown")
        render_score_chart(breakdown)

        st.subheader("🎯 Skill Match Ratio")
        render_skill_ratio(breakdown, jd_skills)

    with right:
        st.subheader("🧩 Skills Extracted")
        with st.container(border=True):
            st.markdown("**From Resume**")
            st.write(", ".join(resume_skills) if resume_skills else "_None detected_")
            st.markdown("**From Job Description**")
            st.write(", ".join(jd_skills) if jd_skills else "_None detected_")

        st.subheader("✅ Matched / ❌ Missing")
        d = breakdown.details["skill"]
        with st.container(border=True):
            st.markdown(f"**Matched ({len(d['matched'])}):** {', '.join(d['matched']) or '—'}")
            st.markdown(f"**Missing ({len(d['missing'])}):** {', '.join(d['missing']) or '—'}")
            if d["critical_missing"]:
                st.error(f"Critical missing: {', '.join(d['critical_missing'])}")

        st.subheader("📅 Experience Detected")
        with st.container(border=True):
            st.write(f"Resume: **{resume_years:.1f} years**")
            st.write(f"JD requires: **{jd_years:.1f} years**" if jd_years else "JD requires: _not specified_")

    st.markdown("---")

    # ---- Component cards ----
    st.subheader("🔍 Component Details")
    cols = st.columns(5)
    items = [
        ("Skill Match", breakdown.skill_match, WEIGHTS["skill_match"]),
        ("Semantic", breakdown.semantic, WEIGHTS["semantic"]),
        ("Keywords", breakdown.keyword, WEIGHTS["keyword"]),
        ("Experience", breakdown.experience, WEIGHTS["experience"]),
        ("Quality", breakdown.quality, WEIGHTS["quality"]),
    ]
    for col, (name, score, mx) in zip(cols, items):
        with col:
            st.metric(f"{color_for(score, mx)} {name}", f"{score:.1f}/{mx}")

    with st.expander("🔑 Keyword analysis"):
        kw = breakdown.details["keyword"]
        st.write(f"Coverage: **{kw['coverage'] * 100:.1f}%**")
        st.markdown(f"**Matched keywords:** {', '.join(kw.get('matched_keywords', [])) or '—'}")
        st.markdown(f"**Missing keywords:** {', '.join(kw.get('missing_keywords', [])) or '—'}")

    with st.expander("📈 Resume quality signals"):
        q = breakdown.details["quality"]
        st.write(f"Projects section: {'✅' if q['has_projects'] else '❌'}")
        st.write(f"Education section: {'✅' if q['has_education'] else '❌'}")
        st.write(f"Action verbs used ({len(q['action_verbs'])}): {', '.join(q['action_verbs']) or '—'}")
        st.write(f"Quantified achievements detected: {q['quantified_count']}")

    with st.expander("📄 Resume preview"):
        st.text_area("Extracted text", value=resume_text, height=300)

    # ---- Suggestions ----
    st.subheader("💡 Suggestions to Improve")
    for tip in feedback:
        st.write(tip)

    # ---- Download report ----
    report = build_report(breakdown, resume_skills, jd_skills, feedback)
    st.download_button(
        "⬇️ Download Detailed Report (TXT)",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
