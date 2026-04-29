import re
import math
import heapq
import importlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..resume_processing.resume_parser import ParsedResume, EDUCATION_ORDINAL
from ..decision_automation.expert_flags import assign_expert_flags
from ..business_optimization.ga_optimizer import get_optimized_weights


_sbert_model = None
_sbert_load_attempted = False


def get_sbert_model():
    global _sbert_model, _sbert_load_attempted

    if _sbert_load_attempted:
        return _sbert_model

    _sbert_load_attempted = True
    try:
        print("[ICRS] Loading SBERT model (all-MiniLM-L6-v2)...")
        sentence_transformers = importlib.import_module("sentence_transformers")
        _sbert_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        print("[ICRS] SBERT model loaded.")
    except Exception as exc:
        _sbert_model = None
        print(f"[ICRS] SBERT unavailable, using lexical fallback. {exc}")

    return _sbert_model


def _cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
    ))


def _token_similarity(text_a, text_b):
    tokens_a = set(re.findall(r"[a-z0-9+#.]+", (text_a or "").lower()))
    tokens_b = set(re.findall(r"[a-z0-9+#.]+", (text_b or "").lower()))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _soft_skill_match(skill_a, skill_b, domain_context=""):
    """Match soft skills and domain-level concepts with expanded understanding."""
    skill_a_lower = (skill_a or "").lower()
    skill_b_lower = (skill_b or "").lower()
    
    # Semantic equivalence mapping for soft skills and domains
    equivalences = {
        "problem solving": ["analytical thinking", "problem-solving", "critical thinking", "troubleshooting"],
        "communication": ["oral communication", "written communication", "interpersonal", "stakeholder management"],
        "leadership": ["team lead", "management", "people management", "team management"],
        "banking": ["financial services", "financial domain", "treasury", "retail banking", "corporate banking", "investment banking"],
        "finance": ["financial services", "accounting", "treasury", "investment"],
        "technical": ["technical skills", "technical expertise", "engineering", "programming"],
        "data science": ["data analytics", "machine learning", "statistics", "data analysis"],
        "testing": ["qa", "quality assurance", "test automation", "manual testing"],
        "project management": ["pm", "agile", "scrum", "waterfall", "prince2"],
    }
    
    # Check direct token overlap first
    if skill_a_lower == skill_b_lower:
        return 1.0
    
    # Check if they share significant tokens
    tokens_a = set(skill_a_lower.split())
    tokens_b = set(skill_b_lower.split())
    if tokens_a & tokens_b:
        return 0.75
    
    # Check equivalence mappings
    for canonical, variants in equivalences.items():
        canonical_match_a = canonical in skill_a_lower
        canonical_match_b = canonical in skill_b_lower
        variant_matches_a = any(v in skill_a_lower for v in variants)
        variant_matches_b = any(v in skill_b_lower for v in variants)
        
        if (canonical_match_a or variant_matches_a) and (canonical_match_b or variant_matches_b):
            return 0.80
    
    return 0.0


@dataclass
class JobDescription:
    title: str = ""
    description: str = ""
    required_skills: list = field(default_factory=list)
    min_experience_years: float = 0.0
    required_education: str = "Bachelors"
    max_notice_period_days: int = 90


@dataclass
class DimensionScore:
    dimension: str
    score: float
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class CandidateRanking:
    name: str
    email: str
    overall_score: float
    dimension_scores: list = field(default_factory=list)
    rank: int = 0
    justification: str = ""
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = ""
    job_titles: list = field(default_factory=list)
    notice_period_days: int = 0
    is_eligible: bool = True
    eligibility_reason: str = ""
    eligibility_trace: str = ""
    expert_flags: list = field(default_factory=list)
    flags_trace: str = ""
    ga_category: str = ""
    ga_weights: dict = field(default_factory=dict)
    reasoning_chain: str = ""


# ═══════════════════════════════════════════════════════════════
# LOAD APRIORI RULES FROM JSON
# ═══════════════════════════════════════════════════════════════

def _load_apriori_rules():
    """Load apriori rules from JSON configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rules_file = os.path.join(current_dir, "apriori_rules.json")
    
    try:
        with open(rules_file, 'r') as f:
            config = json.load(f)
        
        # Convert to format used by scoring engine
        apriori_rules = {}
        for rule in config.get("rules", []):
            antecedent = frozenset(rule["antecedent"])
            consequent = rule["consequent"]
            confidence = rule["confidence"]
            lift = rule["lift"]
            apriori_rules[antecedent] = (consequent, confidence, lift)
        
        apriori_bonus = config.get("configuration", {}).get("apriori_bonus_per_implied_skill", 0.03)
        return apriori_rules, apriori_bonus
    except Exception as e:
        print(f"[ICRS] Warning: Could not load apriori_rules.json: {e}")
        print("[ICRS] Using empty rules fallback.")
        return {}, 0.03


APRIORI_RULES, APRIORI_BONUS = _load_apriori_rules()


# ═══════════════════════════════════════════════════════════════
# D1: Technical Skills (CALIBRATED to 40-95 range)
# ═══════════════════════════════════════════════════════════════
def score_technical_skills(resume, jd):
    resume_skills_set = set(s.lower() for s in resume.skills)
    jd_skills_set = set(s.lower() for s in jd.required_skills)
    sbert_model = get_sbert_model()

    if not jd_skills_set:
        return 0.70, "No specific skills in JD — neutral.", [], []

    matched = resume_skills_set & jd_skills_set
    missing = jd_skills_set - resume_skills_set

    # Keyword match
    keyword_score = len(matched) / len(jd_skills_set)

    # Level 1: Soft skill / domain equivalence matching (lexical)
    soft_skill_matched = set()
    for jd_skill in missing:
        for resume_skill in resume_skills_set:
            equiv_score = _soft_skill_match(jd_skill, resume_skill)
            if equiv_score >= 0.75:
                soft_skill_matched.add(jd_skill)
                missing.discard(jd_skill)
                break

    # Level 2: SBERT semantic match (0.60 threshold for broader matching)
    semantic_matched = set()
    if sbert_model and missing and resume_skills_set:
        unmatched_list = list(missing)
        resume_list = list(resume_skills_set)
        jd_embs = sbert_model.encode(unmatched_list)
        res_embs = sbert_model.encode(resume_list)
        for j, jd_emb in enumerate(jd_embs):
            sims = [_cosine_similarity(jd_emb, resume_emb) for resume_emb in res_embs]
            if sims and max(sims) >= 0.60:  # Lowered threshold from 0.65 to 0.60
                semantic_matched.add(unmatched_list[j])
                missing.discard(unmatched_list[j])

    # Level 3: SBERT match against PHRASES from resume's raw_text.
    # Catches latent skills that the keyword extractor missed — e.g.
    # 'banking' implied by 'Bank of America', 'communication' implied by
    # 'communicating with stakeholders'.
    latent_matched = set()
    if sbert_model and missing and resume.raw_text:
        # Sentences/clauses from the raw text
        phrases = [p.strip() for p in re.split(r"[.\n;:]+", resume.raw_text)
                   if 8 < len(p.strip()) < 250]
        if phrases:
            phrases = phrases[:30]  # cap for efficiency
            phrase_embs = sbert_model.encode(phrases)
            still_missing = list(missing)
            for jd_skill in still_missing:
                jd_emb = sbert_model.encode([jd_skill])[0]
                sims = [_cosine_similarity(jd_emb, ph_emb) for ph_emb in phrase_embs]
                if sims and max(sims) >= 0.55:
                    latent_matched.add(jd_skill)
                    missing.discard(jd_skill)

    total_matched = len(matched) + len(soft_skill_matched) + len(semantic_matched) + len(latent_matched)
    combined_score = total_matched / len(jd_skills_set)

    # Apriori bonus
    apriori_bonus = 0.0
    implied = []
    for antecedent, (consequent, conf, lift) in APRIORI_RULES.items():
        if antecedent.issubset(resume_skills_set) and consequent in jd_skills_set:
            if consequent not in resume_skills_set:
                apriori_bonus += APRIORI_BONUS
                implied.append(f"{consequent}({conf:.0%})")

    # Overall JD-resume relevance boost (prevents harsh scores)
    relevance_boost = 0.0
    if resume.raw_text and jd.description:
        if sbert_model:
            jd_emb = np.array(sbert_model.encode(jd.description[:500])).flatten()
            res_emb = np.array(sbert_model.encode(resume.raw_text[:1000])).flatten()
            overall_sim = _cosine_similarity(jd_emb, res_emb)
        else:
            overall_sim = _token_similarity(jd.description, resume.raw_text)
        relevance_boost = max(0, overall_sim) * 0.25

    # Final: 50% skills + relevance + apriori
    final_score = (combined_score * 0.50) + relevance_boost + apriori_bonus

    # Floor at 0.40 — aligns with GT min of 40%
    final_score = max(0.40, min(1.0, final_score))

    explanation = (
        f"Skills: {len(matched)} keyword + {len(soft_skill_matched)} semantic equiv + "
        f"{len(semantic_matched)} SBERT + {len(latent_matched)} latent / "
        f"{len(jd_skills_set)} required. Relevance boost: +{relevance_boost*100:.0f}%, "
        f"Apriori: {len(implied)} implied."
    )
    matched_union = matched | soft_skill_matched | semantic_matched | latent_matched
    if matched_union:
        explanation += f" Matched: {list(matched_union)[:5]}."

    return final_score, explanation, sorted(matched_union), sorted(missing)


# ═══════════════════════════════════════════════════════════════
# D2: Experience (bell curve, min 50% for juniors)
# ═══════════════════════════════════════════════════════════════
def score_experience(resume, jd):
    actual = resume.experience_years
    required = max(jd.min_experience_years, 1)

    if actual <= 0:
        return 0.50, f"No experience detected (required: {required:.0f}y)."

    ratio = actual / required
    if ratio <= 1.0:
        # Below: lenient curve (min 50%)
        score = 0.50 + (ratio ** 0.7) * 0.50
    elif ratio <= 1.5:
        score = 1.0 - 0.05 * (ratio - 1.0)
    elif ratio <= 2.5:
        score = 0.95 - 0.10 * (ratio - 1.5)
    else:
        score = max(0.55, 0.85 - 0.10 * (ratio - 2.5))

    score = max(0.0, min(1.0, score))

    explanation = f"{actual:.0f}y / required {required:.0f}y (ratio {ratio:.1f}x). "
    if ratio < 0.5:
        explanation += "Below requirement — partial penalty."
    elif ratio < 1.0:
        explanation += "Close to requirement."
    elif ratio <= 1.5:
        explanation += "Optimal range."
    elif ratio <= 2.5:
        explanation += "Above requirement."
    else:
        explanation += "Overqualified — flight risk."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D3: Education (gradient — no hard fail)
# ═══════════════════════════════════════════════════════════════
def score_education(resume, jd):
    candidate_level = EDUCATION_ORDINAL.get(resume.education_level, 0)
    required_level = EDUCATION_ORDINAL.get(jd.required_education, 3)

    if required_level == 0:
        return 0.85, "No education requirement."

    if candidate_level >= required_level:
        score = 1.0
    elif candidate_level == required_level - 1:
        score = 0.75
    elif candidate_level == required_level - 2:
        score = 0.55
    else:
        score = 0.35

    explanation = (
        f"Candidate: {resume.education_level} (L{candidate_level}), "
        f"Required: {jd.required_education} (L{required_level}). "
    )
    if candidate_level >= required_level:
        explanation += "Meets requirement."
    else:
        explanation += f"Below by {required_level - candidate_level}."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D4: Availability (lenient tiers)
# ═══════════════════════════════════════════════════════════════
def score_availability(resume, jd):
    days = resume.notice_period_days
    max_ok = jd.max_notice_period_days

    if days == 0:
        score, tier = 1.0, "Immediately available"
    elif days <= 14:
        score, tier = 0.95, "Within 2 weeks"
    elif days <= 30:
        score, tier = 0.90, "Within 1 month"
    elif days <= 60:
        score, tier = 0.85, "Within 2 months"
    elif days <= 90:
        score, tier = 0.80, "Within 3 months"
    else:
        score, tier = 0.65, f"{days} days"

    if days > max_ok:
        score *= 0.85

    explanation = f"Notice: {days}d ({tier})."
    if days <= max_ok:
        explanation += f" Within ≤{max_ok}d limit."

    return score, explanation


# ═══════════════════════════════════════════════════════════════
# D5: Miscellaneous (SBERT title + relevance, floor 50%)
# ═══════════════════════════════════════════════════════════════
def score_miscellaneous(resume, jd):
    components = []
    sbert_model = get_sbert_model()

    if resume.job_titles and jd.title:
        if sbert_model:
            all_titles = resume.job_titles + [jd.title]
            embs = sbert_model.encode(all_titles)
            jd_emb = np.array(embs[-1])
            best_sim, best_title = 0.0, ""
            for i, title in enumerate(resume.job_titles):
                title_emb = np.array(embs[i])
                sim = _cosine_similarity(title_emb, jd_emb)
                if sim > best_sim:
                    best_sim, best_title = sim, title
        else:
            best_title = max(
                resume.job_titles,
                key=lambda title: _token_similarity(title, jd.title),
                default="",
            )
            best_sim = _token_similarity(best_title, jd.title)

        components.append(("title", max(0.0, best_sim), best_title))
    else:
        components.append(("title", 0.55, "none"))

    summary = resume.summary or resume.raw_text[:500]
    if summary and jd.description:
        if sbert_model:
            embs = sbert_model.encode([summary, jd.description])
            e0, e1 = np.array(embs[0]), np.array(embs[1])
            rel = _cosine_similarity(e0, e1)
        else:
            rel = _token_similarity(summary, jd.description)
        components.append(("relevance", max(0.0, rel), ""))
    else:
        components.append(("relevance", 0.55, ""))

    title_score = components[0][1]
    rel_score = components[1][1]

    final = max(0.50, 0.5 * title_score + 0.5 * rel_score)

    explanation = (
        f"Title: {title_score:.0%} ('{components[0][2]}'). "
        f"Relevance: {rel_score:.0%}."
    )
    return min(1.0, final), explanation


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def rank_candidates(resumes, jd, custom_weights=None, eligibility_results=None):
    """Score and rank resumes that have already passed eligibility."""
    sbert_model = get_sbert_model()

    if eligibility_results is None:
        eligibility_results = [None] * len(resumes)
    elif len(eligibility_results) != len(resumes):
        raise ValueError("eligibility_results must align 1:1 with resumes")

    if custom_weights:
        weights = custom_weights
        ga_category = "custom"
    else:
        weights, ga_category = get_optimized_weights(jd.title, jd.description)

    pq = []

    for idx, resume in enumerate(resumes):
        chain_parts = []
        eligibility = eligibility_results[idx]

        if eligibility is not None:
            if not eligibility.is_eligible:
                raise ValueError("rank_candidates received a non-eligible resume")
            chain_parts.append(eligibility.reasoning_trace)

        # Step 2: Expert Flags
        flag_result = assign_expert_flags(
            resume=resume,
            jd_title=jd.title,
            jd_min_experience=jd.min_experience_years,
            jd_required_education=jd.required_education,
            jd_skills=jd.required_skills,
            jd_text=jd.description,
        )
        chain_parts.append(flag_result.reasoning_trace)

        # Step 3: 5-Dimensional Scoring
        dim_scores = []

        d1_score, d1_expl, matched, missing = score_technical_skills(resume, jd)
        dim_scores.append(DimensionScore(
            "Technical Skills", round(d1_score, 3),
            weights["technical_skills"],
            round(d1_score * weights["technical_skills"], 3), d1_expl))

        d2_score, d2_expl = score_experience(resume, jd)
        dim_scores.append(DimensionScore(
            "Experience", round(d2_score, 3),
            weights["experience"],
            round(d2_score * weights["experience"], 3), d2_expl))

        d3_score, d3_expl = score_education(resume, jd)
        dim_scores.append(DimensionScore(
            "Education", round(d3_score, 3),
            weights["education"],
            round(d3_score * weights["education"], 3), d3_expl))

        d4_score, d4_expl = score_availability(resume, jd)
        dim_scores.append(DimensionScore(
            "Availability", round(d4_score, 3),
            weights["availability"],
            round(d4_score * weights["availability"], 3), d4_expl))

        d5_score, d5_expl = score_miscellaneous(resume, jd)
        dim_scores.append(DimensionScore(
            "Miscellaneous", round(d5_score, 3),
            weights["miscellaneous"],
            round(d5_score * weights["miscellaneous"], 3), d5_expl))

        # Apply flag modifiers
        dim_map = {
            "technical_skills": "Technical Skills",
            "experience": "Experience",
            "education": "Education",
            "availability": "Availability",
            "miscellaneous": "Miscellaneous",
        }
        for flag in flag_result.flags:
            target = dim_map.get(flag.dimension_affected, "")
            for ds in dim_scores:
                if ds.dimension == target:
                    ds.score = max(0, min(1.0, ds.score + flag.score_modifier))
                    ds.weighted_score = round(ds.score * ds.weight, 3)
                    ds.explanation += f" [{flag.flag_name}: {flag.score_modifier:+.0%}]"
                    break

        # Overall score
        overall = sum(ds.weighted_score for ds in dim_scores) * 100
        overall = max(0, min(100, round(overall, 1)))

        chain_parts.append(f"  SCORING: {overall}/100 (category: {ga_category})")
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
            eligibility_trace=eligibility.reasoning_trace if eligibility else "",
            expert_flags=[{
                "name": f.flag_name,
                "value": f.flag_value,  # T/F per the 6-flag spec
                "type": f.flag_type,
                "modifier": f.score_modifier,
                "reason": f.reason,
            } for f in flag_result.flags],
            flags_trace=flag_result.reasoning_trace,
            ga_category=ga_category, ga_weights=weights,
            reasoning_chain="\n".join(chain_parts),
        )

        heapq.heappush(pq, (-overall, idx, ranking))

    # Extract ranked
    results = []
    rank = 1
    while pq:
        _, _, ranking = heapq.heappop(pq)
        ranking.rank = rank
        results.append(ranking)
        rank += 1

    return results


def _generate_justification(resume, dim_scores, overall, flag_result):
    strengths = [f"{ds.dimension} ({ds.score:.0%})" for ds in dim_scores if ds.score >= 0.80]
    weaknesses = [f"{ds.dimension} ({ds.score:.0%})" for ds in dim_scores if ds.score < 0.50]

    parts = [f"{resume.name} scores {overall:.1f}/100."]
    if strengths:
        parts.append(f"Strengths: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Areas to develop: {', '.join(weaknesses)}.")
    if flag_result.bonus_flags:
        parts.append(f"Bonuses: {', '.join(f.flag_name for f in flag_result.bonus_flags)}.")
    if flag_result.penalty_flags:
        parts.append(f"Flags: {', '.join(f.flag_name for f in flag_result.penalty_flags)}.")

    return " ".join(parts)


def parse_job_description(text: str, title: str = "") -> JobDescription:
    from resume_processing.resume_parser import extract_skills, extract_education_level, EDUCATION_ORDINAL

    skills = extract_skills(text)
    exp_match = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    min_exp = float(exp_match[0]) if exp_match else 3.0

    # Use the strict, word-boundary-aware extractor — same logic as for resumes.
    # Default to Bachelors if nothing detected (most professional roles assume this).
    detected_level = extract_education_level(text)
    edu_level = detected_level if detected_level != "Unknown" else "Bachelors"

    return JobDescription(
        title=title or "Software Engineer",
        description=text, required_skills=skills,
        min_experience_years=min_exp,
        required_education=edu_level, max_notice_period_days=90,
    )
