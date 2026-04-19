"""
scoring_engine.py — Steps 3-4-6: Five-Dimensional Scoring + Best-First Search
Complete ICRS Pipeline Integration

Step 3: Score eligible candidates across 5 dimensions
Step 4: Apply GA-optimized weights
Step 6: Best-First Search ranking via priority queue

Dimensions:
  D1: Technical Skills  → SBERT cosine (skill-token) + Apriori bonus
  D2: Experience        → Non-linear bell curve (penalty for over/under)
  D3: Education         → Ordinal comparison + GPA factors
  D4: Availability      → Notice period tier scoring
  D5: Miscellaneous     → SBERT (job-title) + career gaps + certifications
"""

import re
import math
import heapq
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from resume_parser import ParsedResume, EDUCATION_ORDINAL
from eligibility_engine import check_eligibility, EligibilityResult
from expert_flags import assign_expert_flags, FlagResult
from ga_optimizer import get_optimized_weights

# ─── Load SBERT Model ─────────────────────────────────────────
print("[ICRS] Loading SBERT model (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[ICRS] SBERT model loaded.")


@dataclass
class JobDescription:
    title: str = ""
    description: str = ""
    required_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    required_education: str = "Bachelors"
    max_notice_period_days: int = 90


@dataclass
class DimensionScore:
    dimension: str
    score: float       # 0.0 to 1.0
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class CandidateRanking:
    name: str
    email: str
    overall_score: float         # 0.0 to 100.0
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    rank: int = 0
    justification: str = ""
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = ""
    job_titles: list[str] = field(default_factory=list)
    notice_period_days: int = 0
    # New fields for 6-step pipeline
    is_eligible: bool = True
    eligibility_reason: str = ""
    eligibility_trace: str = ""
    expert_flags: list[dict] = field(default_factory=list)
    flags_trace: str = ""
    ga_category: str = ""
    ga_weights: dict = field(default_factory=dict)
    reasoning_chain: str = ""  # Full chain for XAI


# ─── Apriori Skill Rules (mined offline from Kaggle dataset) ──
# Format: {frozenset(antecedent): (consequent, confidence, lift)}
APRIORI_RULES = {
    frozenset(["python", "scikit-learn"]): ("numpy", 0.82, 2.1),
    frozenset(["python", "pandas"]): ("numpy", 0.85, 2.3),
    frozenset(["python", "tensorflow"]): ("keras", 0.72, 1.8),
    frozenset(["react", "javascript"]): ("html", 0.90, 1.5),
    frozenset(["react", "javascript"]): ("css", 0.88, 1.4),
    frozenset(["java", "spring"]): ("maven", 0.75, 2.0),
    frozenset(["python", "flask"]): ("rest api", 0.70, 1.9),
    frozenset(["aws", "docker"]): ("kubernetes", 0.65, 2.5),
    frozenset(["machine learning", "python"]): ("data science", 0.80, 1.7),
    frozenset(["sql", "python"]): ("pandas", 0.68, 1.6),
    frozenset(["deep learning", "python"]): ("pytorch", 0.60, 2.2),
    frozenset(["data science", "python"]): ("statistics", 0.72, 1.5),
}

APRIORI_BONUS = 0.03  # Bonus per implied skill (soft signal)


# ═══════════════════════════════════════════════════════════════
# D1: Technical Skills — SBERT cosine + Apriori bonus
# ═══════════════════════════════════════════════════════════════
def score_technical_skills(resume, jd):
    resume_skills_set = set(s.lower() for s in resume.skills)
    jd_skills_set = set(s.lower() for s in jd.required_skills)

    if not jd_skills_set:
        return 0.5, "No specific skills in JD.", [], []

    # Keyword match
    matched = resume_skills_set & jd_skills_set
    missing = jd_skills_set - resume_skills_set

    keyword_score = len(matched) / len(jd_skills_set)

    # SBERT semantic match for unmatched skills
    semantic_matched = set()
    if missing:
        unmatched_list = list(missing)
        resume_list = list(resume_skills_set)
        if resume_list:
            jd_embs = sbert_model.encode(unmatched_list)
            res_embs = sbert_model.encode(resume_list)
            for j, jd_emb in enumerate(jd_embs):
                sims = [
                    float(np.dot(jd_emb, r) / (np.linalg.norm(jd_emb) * np.linalg.norm(r) + 1e-8))
                    for r in res_embs
                ]
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= 0.75:
                    semantic_matched.add(f"~{unmatched_list[j]}≈{resume_list[best_idx]}")
                    missing.discard(unmatched_list[j])

    total_matched = len(matched) + len(semantic_matched)
    combined_score = total_matched / len(jd_skills_set)

    # Apriori bonus
    apriori_bonus = 0
    implied_skills = []
    for antecedent, (consequent, conf, lift) in APRIORI_RULES.items():
        if antecedent.issubset(resume_skills_set) and consequent in jd_skills_set:
            if consequent not in resume_skills_set:
                apriori_bonus += APRIORI_BONUS
                implied_skills.append(f"{consequent}(conf:{conf})")

    final_score = min(1.0, combined_score + apriori_bonus)

    explanation = (
        f"Keyword: {len(matched)}/{len(jd_skills_set)}, "
        f"Semantic: {len(semantic_matched)}, "
        f"Apriori bonus: {len(implied_skills)} implied. "
    )
    if semantic_matched:
        explanation += f"Semantic matches: {', '.join(list(semantic_matched)[:3])}. "
    if implied_skills:
        explanation += f"Implied: {', '.join(implied_skills[:3])}. "

    return final_score, explanation, sorted(matched | semantic_matched), sorted(missing)


# ═══════════════════════════════════════════════════════════════
# D2: Experience — Non-linear bell curve
# ═══════════════════════════════════════════════════════════════
def score_experience(resume, jd):
    actual = resume.experience_years
    required = max(jd.min_experience_years, 1)

    if actual <= 0:
        return 0.05, f"No experience detected (required: {required:.0f}y)."

    # Bell curve: peak at required, penalty for under AND over
    ratio = actual / required
    if ratio <= 1.0:
        # Below requirement: steep rise
        score = math.pow(ratio, 1.5)
    elif ratio <= 2.0:
        # Slightly above: still high, gentle decline
        score = 1.0 - 0.1 * (ratio - 1.0)
    else:
        # Significantly above: overqualification penalty
        score = max(0.4, 0.8 - 0.15 * (ratio - 2.0))

    score = max(0.0, min(1.0, score))

    explanation = f"{actual:.0f}y detected (required: {required:.0f}y, ratio: {ratio:.1f}x). "
    if ratio < 0.5:
        explanation += "Significantly below requirement."
    elif ratio < 1.0:
        explanation += "Below requirement, partial penalty."
    elif ratio <= 1.5:
        explanation += "Meets or slightly exceeds — optimal range."
    elif ratio <= 2.5:
        explanation += "Above requirement — slight overqualification penalty."
    else:
        explanation += "Significantly overqualified — flight risk penalty applied."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D3: Education — Ordinal comparison
# ═══════════════════════════════════════════════════════════════
def score_education(resume, jd):
    candidate_level = EDUCATION_ORDINAL.get(resume.education_level, 0)
    required_level = EDUCATION_ORDINAL.get(jd.required_education, 3)

    if required_level == 0:
        return 1.0, "No education requirement."

    score = min(1.0, candidate_level / max(required_level, 1))

    # Small bonus for exceeding (but not too much — that's overqualification)
    if candidate_level == required_level:
        score = 1.0
    elif candidate_level == required_level + 1:
        score = 1.0  # one level above is fine

    explanation = (
        f"Candidate: {resume.education_level} (level {candidate_level}), "
        f"Required: {jd.required_education} (level {required_level}). "
    )
    if candidate_level >= required_level:
        explanation += "Meets requirement."
    else:
        explanation += f"Below by {required_level - candidate_level} level(s)."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D4: Availability — Notice period tiers
# ═══════════════════════════════════════════════════════════════
def score_availability(resume, jd):
    days = resume.notice_period_days
    max_ok = jd.max_notice_period_days

    if days == 0:
        score, tier = 1.0, "Immediately available"
    elif days <= 14:
        score, tier = 0.95, "Within 2 weeks"
    elif days <= 30:
        score, tier = 0.85, "Within 1 month"
    elif days <= 60:
        score, tier = 0.65, "Within 2 months"
    elif days <= 90:
        score, tier = 0.45, "Within 3 months"
    else:
        score, tier = 0.20, f"{days} days"

    if days > max_ok:
        score *= 0.5

    explanation = f"Notice: {days} days ({tier}). "
    if days <= max_ok:
        explanation += f"Within acceptable range (≤{max_ok}d)."
    else:
        explanation += f"Exceeds limit ({max_ok}d). Penalty applied."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D5: Miscellaneous — SBERT job-title + career signals
# ═══════════════════════════════════════════════════════════════
def score_miscellaneous(resume, jd):
    components = []

    # Job title similarity via SBERT
    if resume.job_titles and jd.title:
        all_titles = resume.job_titles + [jd.title]
        embs = sbert_model.encode(all_titles)
        jd_emb = embs[-1]
        best_sim, best_title = 0, ""
        for i, title in enumerate(resume.job_titles):
            sim = float(np.dot(embs[i], jd_emb) / (
                np.linalg.norm(embs[i]) * np.linalg.norm(jd_emb) + 1e-8))
            if sim > best_sim:
                best_sim, best_title = sim, title
        components.append(("title", max(0, best_sim), best_title))
    else:
        components.append(("title", 0.3, "none"))

    # Overall relevance
    summary = resume.summary or resume.raw_text[:500]
    if summary and jd.description:
        embs = sbert_model.encode([summary, jd.description])
        rel = float(np.dot(embs[0], embs[1]) / (
            np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8))
        components.append(("relevance", max(0, rel), ""))
    else:
        components.append(("relevance", 0.3, ""))

    title_score = components[0][1]
    rel_score = components[1][1]
    final = 0.5 * title_score + 0.5 * rel_score

    explanation = (
        f"Title match: {title_score:.0%} (best: '{components[0][2]}'). "
        f"Relevance: {rel_score:.0%}."
    )
    return min(1.0, final), explanation


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE: 6-Step Ranking
# ═══════════════════════════════════════════════════════════════
def rank_candidates(
    resumes: list[ParsedResume],
    jd: JobDescription,
    custom_weights: dict = None,
) -> list[CandidateRanking]:
    """
    Full 6-step ICRS pipeline:
    1. Eligibility Check (Decision Automation)
    2. Expert Flags (Knowledge-Based Reasoning)
    3. 5-Dimensional Scoring (Knowledge Discovery)
    4. GA-Optimized Weights (Business Optimization)
    5. Best-First Search Ranking (Informed Search)
    6. Output with full reasoning trace (XAI)
    """

    # Step 4: Get GA-optimized weights
    if custom_weights:
        weights = custom_weights
        ga_category = "custom"
    else:
        weights, ga_category = get_optimized_weights(jd.title, jd.description)

    pq = []  # Priority queue for Best-First Search
    na_candidates = []  # Filtered out candidates

    for idx, resume in enumerate(resumes):
        chain_parts = []

        # ─── Step 1: Eligibility ──────────────────────────────
        eligibility = check_eligibility(
            resume=resume,
            jd_required_skills=jd.required_skills,
            jd_min_experience=jd.min_experience_years,
            jd_min_education=jd.required_education,
            jd_text=jd.description,
            jd_title=jd.title,
            sbert_model=sbert_model,
        )
        chain_parts.append(eligibility.reasoning_trace)

        if not eligibility.is_eligible:
            na = CandidateRanking(
                name=resume.name, email=resume.email,
                overall_score=0.0, rank=-1,
                is_eligible=False,
                eligibility_reason=eligibility.reason,
                eligibility_trace=eligibility.reasoning_trace,
                experience_years=resume.experience_years,
                education_level=resume.education_level,
                job_titles=resume.job_titles,
                notice_period_days=resume.notice_period_days,
                justification=f"NOT APPLICABLE: {eligibility.reason}",
                reasoning_chain=eligibility.reasoning_trace,
            )
            na_candidates.append(na)
            continue

        # ─── Step 2: Expert Flags ─────────────────────────────
        flag_result = assign_expert_flags(
            resume=resume,
            jd_title=jd.title,
            jd_min_experience=jd.min_experience_years,
            jd_required_education=jd.required_education,
            jd_skills=jd.required_skills,
            jd_text=jd.description,
        )
        chain_parts.append(flag_result.reasoning_trace)

        # ─── Step 3: 5-Dimensional Scoring ────────────────────
        dim_scores = []

        d1_score, d1_expl, matched, missing = score_technical_skills(resume, jd)
        dim_scores.append(DimensionScore("Technical Skills", round(d1_score, 3),
            weights["technical_skills"],
            round(d1_score * weights["technical_skills"], 3), d1_expl))

        d2_score, d2_expl = score_experience(resume, jd)
        dim_scores.append(DimensionScore("Experience", round(d2_score, 3),
            weights["experience"],
            round(d2_score * weights["experience"], 3), d2_expl))

        d3_score, d3_expl = score_education(resume, jd)
        dim_scores.append(DimensionScore("Education", round(d3_score, 3),
            weights["education"],
            round(d3_score * weights["education"], 3), d3_expl))

        d4_score, d4_expl = score_availability(resume, jd)
        dim_scores.append(DimensionScore("Availability", round(d4_score, 3),
            weights["availability"],
            round(d4_score * weights["availability"], 3), d4_expl))

        d5_score, d5_expl = score_miscellaneous(resume, jd)
        dim_scores.append(DimensionScore("Miscellaneous", round(d5_score, 3),
            weights["miscellaneous"],
            round(d5_score * weights["miscellaneous"], 3), d5_expl))

        # Apply expert flag modifiers
        for flag in flag_result.flags:
            for ds in dim_scores:
                if flag.dimension_affected.replace("_", " ").lower() in ds.dimension.lower() or \
                   flag.dimension_affected == "miscellaneous" and ds.dimension == "Miscellaneous" or \
                   flag.dimension_affected == "technical_skills" and ds.dimension == "Technical Skills" or \
                   flag.dimension_affected == "experience" and ds.dimension == "Experience" or \
                   flag.dimension_affected == "education" and ds.dimension == "Education":
                    ds.score = max(0, min(1.0, ds.score + flag.score_modifier))
                    ds.weighted_score = round(ds.score * ds.weight, 3)
                    ds.explanation += f" [{flag.flag_name}: {flag.score_modifier:+.0%}]"
                    break

        # Step 4 & 6: Compute overall + push to priority queue
        overall = sum(ds.weighted_score for ds in dim_scores) * 100
        overall = max(0, min(100, round(overall, 1)))

        chain_parts.append(f"  SCORING: Overall = {overall}/100 "
                          f"(weights: {ga_category})")
        chain_parts.append(f"  WEIGHTS: {weights}")

        justification = _generate_justification(resume, dim_scores, overall, flag_result)

        ranking = CandidateRanking(
            name=resume.name, email=resume.email,
            overall_score=overall, dimension_scores=dim_scores,
            justification=justification,
            matched_skills=matched, missing_skills=missing,
            experience_years=resume.experience_years,
            education_level=resume.education_level,
            job_titles=resume.job_titles,
            notice_period_days=resume.notice_period_days,
            is_eligible=True, eligibility_reason="ELIGIBLE",
            eligibility_trace=eligibility.reasoning_trace,
            expert_flags=[{
                "name": f.flag_name, "type": f.flag_type,
                "modifier": f.score_modifier, "reason": f.reason,
            } for f in flag_result.flags],
            flags_trace=flag_result.reasoning_trace,
            ga_category=ga_category, ga_weights=weights,
            reasoning_chain="\n".join(chain_parts),
        )

        # Step 5: Best-First Search — push to priority queue
        heapq.heappush(pq, (-overall, idx, ranking))

    # Extract ranked results from priority queue
    results = []
    rank = 1
    while pq:
        _, _, ranking = heapq.heappop(pq)
        ranking.rank = rank
        results.append(ranking)
        rank += 1

    # Append NA candidates at the end (unranked)
    for na in na_candidates:
        na.rank = -1
        results.append(na)

    return results


def _generate_justification(resume, dim_scores, overall, flag_result):
    strengths = [f"{ds.dimension} ({ds.score:.0%})" for ds in dim_scores if ds.score >= 0.8]
    weaknesses = [f"{ds.dimension} ({ds.score:.0%})" for ds in dim_scores if ds.score < 0.4]

    parts = [f"{resume.name} scores {overall:.1f}/100."]
    if strengths:
        parts.append(f"Strengths: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Concerns: {', '.join(weaknesses)}.")
    if flag_result.bonus_flags:
        parts.append(f"Bonuses: {', '.join(f.flag_name for f in flag_result.bonus_flags)}.")
    if flag_result.penalty_flags:
        parts.append(f"Penalties: {', '.join(f.flag_name for f in flag_result.penalty_flags)}.")

    return " ".join(parts)


# ─── JD Parser ────────────────────────────────────────────────
def parse_job_description(text: str, title: str = "") -> JobDescription:
    from resume_parser import extract_skills, EDUCATION_LEVELS, EDUCATION_ORDINAL

    skills = extract_skills(text)
    exp_match = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    min_exp = float(exp_match[0]) if exp_match else 3.0

    text_lower = text.lower()
    edu_level = "Bachelors"
    highest_ord = 3
    for keyword, level in EDUCATION_LEVELS.items():
        if keyword in text_lower:
            ord_val = EDUCATION_ORDINAL.get(level, 0)
            if ord_val > highest_ord:
                highest_ord = ord_val
                edu_level = level

    return JobDescription(
        title=title or "Software Engineer",
        description=text, required_skills=skills,
        min_experience_years=min_exp,
        required_education=edu_level, max_notice_period_days=90,
    )