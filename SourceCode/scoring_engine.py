"""
scoring_engine.py — Five-Dimensional Candidate Scoring Pipeline
Implements the S-Rank ICRS scoring methodology:
  D1: Technical Skills  (SBERT cosine similarity + keyword overlap)
  D2: Experience        (Non-linear logarithmic scaling)
  D3: Education         (Ordinal comparison against JD requirement)
  D4: Availability      (Notice period tier scoring)
  D5: Miscellaneous     (SBERT job-title similarity to JD role)
"""

import math
import heapq
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from resume_parser import ParsedResume, EDUCATION_ORDINAL


# ─── Load SBERT Model ─────────────────────────────────────────
# Using a lightweight model for fast inference
# In production, use 'all-mpnet-base-v2' for better accuracy
print("[ICRS] Loading SBERT model (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[ICRS] SBERT model loaded successfully.")


@dataclass
class JobDescription:
    """Structured representation of a job description."""
    title: str = ""
    description: str = ""
    required_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    required_education: str = "Bachelors"  # minimum education level
    max_notice_period_days: int = 90  # acceptable notice period


@dataclass
class DimensionScore:
    """Score for a single dimension with explanation."""
    dimension: str
    score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class CandidateRanking:
    """Complete ranking result for a candidate."""
    name: str
    email: str
    overall_score: float  # 0.0 to 100.0
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    rank: int = 0
    justification: str = ""
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = ""
    job_titles: list[str] = field(default_factory=list)
    notice_period_days: int = 0


# ─── Default Dimension Weights ────────────────────────────────
# In the full S-Rank system, these are optimized via GA (DEAP)
# with Kendall Tau fitness function. Here we use sensible defaults.
DEFAULT_WEIGHTS = {
    "technical_skills": 0.35,
    "experience": 0.25,
    "education": 0.15,
    "availability": 0.10,
    "miscellaneous": 0.15,
}


# ═══════════════════════════════════════════════════════════════
# DIMENSION 1: Technical Skills Scoring
# Method: SBERT cosine similarity + keyword overlap (Jaccard-like)
# ═══════════════════════════════════════════════════════════════
def score_technical_skills(
    resume: ParsedResume,
    jd: JobDescription,
) -> tuple[float, str, list[str], list[str]]:
    """
    Score technical skills match using two components:
    1. Keyword overlap (hard skills matching)
    2. SBERT semantic similarity (captures related skills)
    """
    resume_skills_set = set(s.lower() for s in resume.skills)
    jd_skills_set = set(s.lower() for s in jd.required_skills)

    if not jd_skills_set:
        # If no specific skills in JD, use SBERT on full text
        jd_skills_set = set()

    # Component 1: Keyword overlap (Jaccard-like)
    matched = resume_skills_set & jd_skills_set
    missing = jd_skills_set - resume_skills_set

    if jd_skills_set:
        keyword_score = len(matched) / len(jd_skills_set)
    else:
        keyword_score = 0.5  # neutral if no specific skills listed

    # Component 2: SBERT semantic similarity
    # Encode resume skills as a single string and compare to JD
    resume_skill_text = ", ".join(resume.skills) if resume.skills else "no specific skills"
    jd_skill_text = ", ".join(jd.required_skills) if jd.required_skills else jd.description

    embeddings = sbert_model.encode([resume_skill_text, jd_skill_text])
    cosine_sim = float(np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
    ))
    semantic_score = max(0, min(1, cosine_sim))

    # Weighted combination: 60% keyword + 40% semantic
    final_score = 0.6 * keyword_score + 0.4 * semantic_score

    explanation = (
        f"Matched {len(matched)}/{len(jd_skills_set)} required skills "
        f"(keyword: {keyword_score:.0%}, semantic: {semantic_score:.0%}). "
    )
    if matched:
        explanation += f"Matched: {', '.join(sorted(matched)[:8])}. "
    if missing:
        explanation += f"Missing: {', '.join(sorted(missing)[:5])}."

    return final_score, explanation, sorted(matched), sorted(missing)


# ═══════════════════════════════════════════════════════════════
# DIMENSION 2: Experience Scoring
# Method: Non-linear logarithmic scaling
# ═══════════════════════════════════════════════════════════════
def score_experience(
    resume: ParsedResume,
    jd: JobDescription,
) -> tuple[float, str]:
    """
    Score experience using non-linear (logarithmic) scaling.
    This captures diminishing returns: the difference between
    1 and 3 years is more significant than 15 vs 17 years.
    
    Formula: score = log(1 + actual) / log(1 + max_expected)
    Capped at 1.0 (meeting or exceeding requirement scores full marks).
    """
    actual = resume.experience_years
    required = max(jd.min_experience_years, 1)
    max_expected = required * 2  # consider up to 2x requirement

    if actual <= 0:
        score = 0.0
    else:
        score = math.log(1 + actual) / math.log(1 + max_expected)
        score = min(1.0, score)

    # Bonus for meeting/exceeding requirement
    if actual >= required:
        score = max(score, 0.7)

    explanation = (
        f"{actual:.0f} years detected (required: {required:.0f}+). "
        f"Log-scaled score: {score:.0%}."
    )
    if actual >= required * 1.5:
        explanation += " Significantly exceeds requirement."
    elif actual >= required:
        explanation += " Meets requirement."
    elif actual >= required * 0.5:
        explanation += " Partially meets requirement."
    else:
        explanation += " Below minimum requirement."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# DIMENSION 3: Education Scoring
# Method: Ordinal comparison against JD requirement
# ═══════════════════════════════════════════════════════════════
def score_education(
    resume: ParsedResume,
    jd: JobDescription,
) -> tuple[float, str]:
    """
    Score education using ordinal level comparison.
    PhD=5, Masters=4, Bachelors=3, Diploma=2, Certificate=1, HighSchool=0
    
    Score = candidate_level / required_level (capped at 1.0)
    Bonus for exceeding requirement.
    """
    candidate_level = EDUCATION_ORDINAL.get(resume.education_level, 0)
    required_level = EDUCATION_ORDINAL.get(jd.required_education, 3)

    if required_level == 0:
        score = 1.0  # no requirement
    else:
        score = min(1.0, candidate_level / required_level)

    # Bonus for exceeding
    if candidate_level > required_level:
        score = min(1.0, score + 0.1)

    explanation = (
        f"Candidate: {resume.education_level} (level {candidate_level}), "
        f"Required: {jd.required_education} (level {required_level}). "
    )
    if candidate_level >= required_level:
        explanation += "Meets or exceeds requirement."
    else:
        explanation += "Below requirement."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# DIMENSION 4: Availability Scoring
# Method: Notice period tier scoring
# ═══════════════════════════════════════════════════════════════
def score_availability(
    resume: ParsedResume,
    jd: JobDescription,
) -> tuple[float, str]:
    """
    Score availability based on notice period tiers:
    - Immediately available (0 days): 1.0
    - Within 2 weeks (≤14 days): 0.9
    - Within 1 month (≤30 days): 0.8
    - Within 2 months (≤60 days): 0.6
    - Within 3 months (≤90 days): 0.4
    - More than 3 months: 0.2
    """
    days = resume.notice_period_days
    max_acceptable = jd.max_notice_period_days

    if days == 0:
        score = 1.0
        tier = "Immediately available"
    elif days <= 14:
        score = 0.9
        tier = "Within 2 weeks"
    elif days <= 30:
        score = 0.8
        tier = "Within 1 month"
    elif days <= 60:
        score = 0.6
        tier = "Within 2 months"
    elif days <= 90:
        score = 0.4
        tier = "Within 3 months"
    else:
        score = 0.2
        tier = f"{days} days notice"

    # Penalty if exceeds max acceptable
    if days > max_acceptable:
        score *= 0.5

    explanation = f"Notice period: {days} days ({tier}). "
    if days <= max_acceptable:
        explanation += f"Within acceptable range (≤{max_acceptable} days)."
    else:
        explanation += f"Exceeds acceptable range ({max_acceptable} days). Penalty applied."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# DIMENSION 5: Miscellaneous Scoring
# Method: SBERT job-title similarity to JD role
# ═══════════════════════════════════════════════════════════════
def score_miscellaneous(
    resume: ParsedResume,
    jd: JobDescription,
) -> tuple[float, str]:
    """
    Score miscellaneous factors:
    - Job title similarity to the role (SBERT cosine)
    - Resume summary relevance to JD (SBERT cosine)
    """
    scores_components = []

    # Component 1: Job title match
    if resume.job_titles and jd.title:
        title_texts = resume.job_titles + [jd.title]
        title_embeddings = sbert_model.encode(title_texts)
        jd_title_emb = title_embeddings[-1]

        best_title_sim = 0
        best_title = ""
        for i, title in enumerate(resume.job_titles):
            sim = float(np.dot(title_embeddings[i], jd_title_emb) / (
                np.linalg.norm(title_embeddings[i]) * np.linalg.norm(jd_title_emb) + 1e-8
            ))
            if sim > best_title_sim:
                best_title_sim = sim
                best_title = title

        scores_components.append(("title_match", max(0, best_title_sim), best_title))
    else:
        scores_components.append(("title_match", 0.3, "No titles found"))

    # Component 2: Summary/overall relevance
    summary_text = resume.summary or resume.raw_text[:500]
    if summary_text and jd.description:
        embs = sbert_model.encode([summary_text, jd.description])
        summary_sim = float(np.dot(embs[0], embs[1]) / (
            np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
        ))
        scores_components.append(("relevance", max(0, summary_sim), ""))
    else:
        scores_components.append(("relevance", 0.3, ""))

    # Weighted average: 50% title match + 50% relevance
    title_score = scores_components[0][1]
    relevance_score = scores_components[1][1]
    final_score = 0.5 * title_score + 0.5 * relevance_score

    explanation = (
        f"Title match: {title_score:.0%} "
        f"(best: '{scores_components[0][2]}'). "
        f"Overall relevance: {relevance_score:.0%}."
    )

    return min(1.0, final_score), explanation


# ═══════════════════════════════════════════════════════════════
# RANKING ENGINE: Best-First Search (Priority Queue)
# ═══════════════════════════════════════════════════════════════
def rank_candidates(
    resumes: list[ParsedResume],
    jd: JobDescription,
    weights: dict[str, float] | None = None,
) -> list[CandidateRanking]:
    """
    Rank candidates using Best-First Search with priority queue.
    
    Each candidate is scored across 5 dimensions, weighted,
    and inserted into a max-heap (priority queue). The queue
    naturally produces candidates in descending order of
    overall score.
    
    This is explicitly Best-First Search (not A*) because
    candidates are scored independently — there's no path cost
    or heuristic to a goal state. Each candidate is a complete
    state evaluated by its priority (aggregate score).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Priority queue: use negative score for max-heap behavior
    pq: list[tuple[float, int, CandidateRanking]] = []

    for idx, resume in enumerate(resumes):
        dim_scores = []

        # D1: Technical Skills
        d1_score, d1_expl, matched, missing = score_technical_skills(resume, jd)
        dim_scores.append(DimensionScore(
            dimension="Technical Skills",
            score=round(d1_score, 3),
            weight=weights["technical_skills"],
            weighted_score=round(d1_score * weights["technical_skills"], 3),
            explanation=d1_expl,
        ))

        # D2: Experience
        d2_score, d2_expl = score_experience(resume, jd)
        dim_scores.append(DimensionScore(
            dimension="Experience",
            score=round(d2_score, 3),
            weight=weights["experience"],
            weighted_score=round(d2_score * weights["experience"], 3),
            explanation=d2_expl,
        ))

        # D3: Education
        d3_score, d3_expl = score_education(resume, jd)
        dim_scores.append(DimensionScore(
            dimension="Education",
            score=round(d3_score, 3),
            weight=weights["education"],
            weighted_score=round(d3_score * weights["education"], 3),
            explanation=d3_expl,
        ))

        # D4: Availability
        d4_score, d4_expl = score_availability(resume, jd)
        dim_scores.append(DimensionScore(
            dimension="Availability",
            score=round(d4_score, 3),
            weight=weights["availability"],
            weighted_score=round(d4_score * weights["availability"], 3),
            explanation=d4_expl,
        ))

        # D5: Miscellaneous
        d5_score, d5_expl = score_miscellaneous(resume, jd)
        dim_scores.append(DimensionScore(
            dimension="Miscellaneous",
            score=round(d5_score, 3),
            weight=weights["miscellaneous"],
            weighted_score=round(d5_score * weights["miscellaneous"], 3),
            explanation=d5_expl,
        ))

        # Overall weighted score (0-100 scale)
        overall = sum(ds.weighted_score for ds in dim_scores) * 100

        # Generate justification
        justification = generate_justification(resume, dim_scores, overall)

        ranking = CandidateRanking(
            name=resume.name,
            email=resume.email,
            overall_score=round(overall, 1),
            dimension_scores=dim_scores,
            matched_skills=matched,
            missing_skills=missing,
            experience_years=resume.experience_years,
            education_level=resume.education_level,
            job_titles=resume.job_titles,
            notice_period_days=resume.notice_period_days,
            justification=justification,
        )

        # Push to priority queue (negative for max-heap)
        heapq.heappush(pq, (-overall, idx, ranking))

    # Extract ranked results
    results = []
    rank = 1
    while pq:
        _, _, ranking = heapq.heappop(pq)
        ranking.rank = rank
        results.append(ranking)
        rank += 1

    return results


def generate_justification(
    resume: ParsedResume,
    dim_scores: list[DimensionScore],
    overall: float,
) -> str:
    """
    Generate a rule-based explanation for the candidate's ranking.
    Uses forward-chaining logic to highlight strengths and weaknesses.
    """
    strengths = []
    weaknesses = []

    for ds in dim_scores:
        if ds.score >= 0.8:
            strengths.append(f"{ds.dimension} ({ds.score:.0%})")
        elif ds.score < 0.4:
            weaknesses.append(f"{ds.dimension} ({ds.score:.0%})")

    parts = [f"{resume.name} scores {overall:.1f}/100 overall."]

    if strengths:
        parts.append(f"Key strengths: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Areas of concern: {', '.join(weaknesses)}.")
    if resume.experience_years >= 5:
        parts.append(f"Brings {resume.experience_years:.0f} years of industry experience.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# SKILL EXTRACTION FROM JD (for auto-parsing job descriptions)
# ═══════════════════════════════════════════════════════════════
def parse_job_description(text: str, title: str = "") -> JobDescription:
    """Parse a job description text into structured format."""
    from resume_parser import extract_skills, EDUCATION_LEVELS, EDUCATION_ORDINAL

    skills = extract_skills(text)

    # Extract experience requirement
    exp_match = re.findall(
        r"(\d{1,2})\+?\s*(?:years?|yrs?)",
        text, re.IGNORECASE
    )
    min_exp = float(exp_match[0]) if exp_match else 3.0

    # Extract education requirement
    text_lower = text.lower()
    edu_level = "Bachelors"  # default
    highest_ord = EDUCATION_ORDINAL.get("Bachelors", 3)
    for keyword, level in EDUCATION_LEVELS.items():
        if keyword in text_lower:
            ord_val = EDUCATION_ORDINAL.get(level, 0)
            if ord_val > highest_ord:
                highest_ord = ord_val
                edu_level = level

    return JobDescription(
        title=title or "Software Engineer",
        description=text,
        required_skills=skills,
        min_experience_years=min_exp,
        required_education=edu_level,
        max_notice_period_days=90,
    )
