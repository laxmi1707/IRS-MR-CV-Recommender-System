"""
eligibility_engine.py — Step 1: Decision Automation Layer
RELAXED Forward Chaining Inference Engine

Aligned with ground truth (JDVsCDRanking.csv shows 130/130 eligible).
Philosophy: "No rejection, only positioning."

Only mark NA in extreme cases:
1. Clearly unrelated profession (Yoga/Chef/Plumber in titles)
2. Specialized degree mismatch (LLB vs BE when JD strictly requires LLB)
3. Education gap of 2+ levels when JD explicitly requires it

All other candidates proceed to scoring. Weaker fit → lower score, still ranked.
"""

import re
from dataclasses import dataclass, field
from resume_parser import ParsedResume


@dataclass
class RuleFiring:
    rule_name: str
    condition: str
    result: str
    details: str


@dataclass
class EligibilityResult:
    is_eligible: bool
    reason: str
    rules_fired: list = field(default_factory=list)
    skill_match_ratio: float = 0.0
    reasoning_trace: str = ""


# Truly unrelated professions — strong filter signals only
UNRELATED_PROFESSION_KEYWORDS = [
    "yoga instructor", "yoga teacher", "fitness trainer", "personal trainer",
    "chef", "sous chef", "head chef", "pastry chef",
    "hairdresser", "beautician", "cosmetologist", "makeup artist",
    "plumber", "electrician", "carpenter", "welder", "mason",
    "gardener", "florist", "landscaper",
    "truck driver", "cab driver", "delivery driver",
    "security guard", "bouncer",
    "waiter", "waitress", "bartender",
]


def _check_unrelated_profession(resume: ParsedResume, jd_text: str):
    """Only flag if resume is from clearly unrelated profession."""
    resume_text_lower = resume.raw_text.lower()
    resume_titles_lower = " ".join(resume.job_titles).lower()
    jd_lower = jd_text.lower()

    # If JD itself is for one of these domains, don't filter
    if any(kw in jd_lower for kw in UNRELATED_PROFESSION_KEYWORDS):
        return False, []

    # Check job titles — strongest signal
    matched_in_titles = [kw for kw in UNRELATED_PROFESSION_KEYWORDS
                         if kw in resume_titles_lower]
    if matched_in_titles:
        return True, matched_in_titles

    # Body text needs 3+ hits to avoid false positives
    matched_in_text = [kw for kw in UNRELATED_PROFESSION_KEYWORDS
                       if kw in resume_text_lower]
    if len(matched_in_text) >= 3:
        return True, matched_in_text

    return False, []


def check_eligibility(
    resume: ParsedResume,
    jd_required_skills: list,
    jd_min_experience: float,
    jd_min_education: str,
    jd_text: str,
    jd_title: str = "",
    sbert_model=None,
) -> EligibilityResult:
    """Relaxed eligibility check — defaults to ELIGIBLE."""
    rules_fired = []
    is_eligible = True
    fail_reason = ""
    skill_ratio = 0.5

    resume_skills_lower = set(s.lower() for s in resume.skills)
    jd_skills_lower = set(s.lower() for s in jd_required_skills)

    # ─── Rule 1: Unrelated Profession ─────────────────────────
    is_unrelated, matched_kw = _check_unrelated_profession(resume, jd_text)
    if is_unrelated:
        rules_fired.append(RuleFiring(
            rule_name="UnrelatedProfessionRule",
            condition=f"Unrelated profession indicators: {matched_kw[:3]}",
            result="FAIL",
            details=f"Resume from clearly unrelated profession: {', '.join(matched_kw[:5])}.",
        ))
        is_eligible = False
        fail_reason = f"Unrelated profession ({', '.join(matched_kw[:2])})"
    else:
        rules_fired.append(RuleFiring(
            rule_name="UnrelatedProfessionRule",
            condition="No unrelated profession keywords",
            result="PASS",
            details="Candidate's professional background is compatible.",
        ))

    # ─── Rule 2: Skill Overlap (ZERO match only) ──────────────
    if is_eligible and jd_skills_lower:
        matched_skills = resume_skills_lower & jd_skills_lower
        semantic_matches = 0

        # Semantic check (lenient threshold)
        if sbert_model and (jd_skills_lower - matched_skills):
            import numpy as np
            unmatched_jd = list(jd_skills_lower - matched_skills)
            resume_skill_list = list(resume_skills_lower)
            if resume_skill_list and unmatched_jd:
                jd_embs = sbert_model.encode(unmatched_jd)
                res_embs = sbert_model.encode(resume_skill_list)
                for j, jd_emb in enumerate(jd_embs):
                    sims = [float(np.dot(jd_emb, r) / (
                        np.linalg.norm(jd_emb) * np.linalg.norm(r) + 1e-8))
                        for r in res_embs]
                    if sims and max(sims) >= 0.65:
                        semantic_matches += 1

        total_matches = len(matched_skills) + semantic_matches
        skill_ratio = total_matches / max(1, len(jd_skills_lower))

        # ONLY reject if ZERO matches AND resume has substantial skills
        if total_matches == 0 and len(resume.skills) >= 3:
            rules_fired.append(RuleFiring(
                rule_name="SkillOverlapRule",
                condition=f"0 matches out of {len(jd_skills_lower)} required",
                result="FAIL",
                details=f"Zero overlap with {len(resume.skills)} listed skills.",
            ))
            is_eligible = False
            fail_reason = f"Zero skill overlap (0/{len(jd_skills_lower)})"
        else:
            rules_fired.append(RuleFiring(
                rule_name="SkillOverlapRule",
                condition=f"{total_matches}/{len(jd_skills_lower)} matched ({skill_ratio:.0%})",
                result="PASS",
                details="Will be scored on D1.",
            ))

    # ─── Rule 3: Experience (always pass) ─────────────────────
    if is_eligible:
        rules_fired.append(RuleFiring(
            rule_name="ExperienceRule",
            condition=f"exp={resume.experience_years}y, required={jd_min_experience}y",
            result="PASS",
            details="No rejection — handled via D2 scoring.",
        ))

    # ─── Rule 4: Education (specialized stream + 2-level gap) ─
    from resume_parser import EDUCATION_ORDINAL
    if is_eligible and jd_min_education:
        candidate_edu_ord = EDUCATION_ORDINAL.get(resume.education_level, 0)
        required_edu_ord = EDUCATION_ORDINAL.get(jd_min_education, 0)

        SPECIALIZED_DEGREES = {
            "llb": "law", "llm": "law",
            "mbbs": "medicine", "md ": "medicine", "bds": "medicine",
            "b.arch": "architecture",
            "bfa": "arts",
        }

        jd_edu_lower = (jd_min_education + " " + jd_text).lower()
        jd_stream = None
        for keyword, stream in SPECIALIZED_DEGREES.items():
            if re.search(r"\b" + re.escape(keyword), jd_edu_lower):
                jd_stream = stream
                break

        candidate_stream = None
        resume_edu_lower = (resume.education_text + " " + resume.education_level).lower()
        for keyword, stream in SPECIALIZED_DEGREES.items():
            if re.search(r"\b" + re.escape(keyword), resume_edu_lower):
                candidate_stream = stream
                break

        if jd_stream and candidate_stream != jd_stream:
            rules_fired.append(RuleFiring(
                rule_name="EducationStreamRule",
                condition=f"JD requires '{jd_stream}', candidate has '{candidate_stream or 'general'}'",
                result="FAIL",
                details=f"Specialized {jd_stream} degree required.",
            ))
            is_eligible = False
            fail_reason = f"Wrong education stream ({candidate_stream or 'general'} vs {jd_stream})"
        elif required_edu_ord > 0 and candidate_edu_ord > 0:
            gap = required_edu_ord - candidate_edu_ord
            if gap >= 2:
                rules_fired.append(RuleFiring(
                    rule_name="EducationLevelRule",
                    condition=f"Gap of {gap} levels below requirement",
                    result="FAIL",
                    details=f"Significant education gap ({resume.education_level} vs {jd_min_education}).",
                ))
                is_eligible = False
                fail_reason = f"Education too low ({resume.education_level} vs {jd_min_education})"
            else:
                rules_fired.append(RuleFiring(
                    rule_name="EducationLevelRule",
                    condition=f"Candidate: {resume.education_level}, Required: {jd_min_education}",
                    result="PASS",
                    details="Education acceptable (within 1 level or higher).",
                ))
        else:
            rules_fired.append(RuleFiring(
                rule_name="EducationLevelRule",
                condition="No strict education requirement",
                result="PASS",
                details="Will score via D3.",
            ))

    # ─── Build Trace ──────────────────────────────────────────
    trace_lines = [f"=== Eligibility Check: {resume.name} ==="]
    for rf in rules_fired:
        trace_lines.append(f"  [{rf.result}] {rf.rule_name}: {rf.condition}")
        trace_lines.append(f"         → {rf.details}")
    if is_eligible:
        trace_lines.append("  VERDICT: ELIGIBLE — proceeds to scoring.")
    else:
        trace_lines.append(f"  VERDICT: NOT APPLICABLE — {fail_reason}")

    return EligibilityResult(
        is_eligible=is_eligible,
        reason="ELIGIBLE" if is_eligible else fail_reason,
        rules_fired=rules_fired,
        skill_match_ratio=skill_ratio,
        reasoning_trace="\n".join(trace_lines),
    )