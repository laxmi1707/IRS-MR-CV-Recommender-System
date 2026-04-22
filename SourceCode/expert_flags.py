"""
expert_flags.py — Step 2: Knowledge-Based Reasoning Layer
Forward Chaining Expert Flag System (aligned with Ground Truth)

Ground truth tracks these flags:
- RELOCATION_FLAG, WORK_VISA_FLAG, CAREER_GAP_FLAG, EDUCATION_GAP_FLAG
- CERTIFICATION_SCORE, OVERQUALIFICATION_FLAG, CAREER_STABILITY_FLAG
- LEADERSHIP_MATCH, EXACT_TITLE_MATCH, FRESH_GRADUATE, UPSKILL_FLAG
"""

import re
from dataclasses import dataclass, field
from resume_parser import ParsedResume, EDUCATION_ORDINAL


@dataclass
class ExpertFlag:
    flag_name: str
    flag_type: str
    score_modifier: float
    dimension_affected: str
    reason: str
    confidence: float


@dataclass
class FlagResult:
    flags: list = field(default_factory=list)
    reasoning_trace: str = ""

    @property
    def total_bonus(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier > 0)

    @property
    def total_penalty(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier < 0)

    @property
    def bonus_flags(self) -> list:
        return [f for f in self.flags if f.flag_type == "BONUS"]

    @property
    def penalty_flags(self) -> list:
        return [f for f in self.flags if f.flag_type == "PENALTY"]


def assign_expert_flags(
    resume: ParsedResume,
    jd_title: str,
    jd_min_experience: float,
    jd_required_education: str,
    jd_skills: list,
    jd_text: str = "",
) -> FlagResult:
    """Forward chaining rule engine — 11+ expert flags."""
    flags = []
    trace_lines = [f"=== Expert Flags: {resume.name} ==="]
    resume_text_lower = resume.raw_text.lower()
    jd_text_lower = jd_text.lower()

    # ─── OVERQUALIFICATION_FLAG ───────────────────────────────
    if jd_min_experience > 0 and resume.experience_years >= jd_min_experience * 2.5:
        flag = ExpertFlag(
            flag_name="OVERQUALIFICATION_FLAG",
            flag_type="PENALTY",
            score_modifier=-0.06,
            dimension_affected="experience",
            reason=f"Has {resume.experience_years:.0f}y vs required {jd_min_experience:.0f}y. Flight risk.",
            confidence=0.7,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] OVERQUALIFICATION_FLAG: {flag.reason}")

    candidate_edu_ord = EDUCATION_ORDINAL.get(resume.education_level, 0)
    required_edu_ord = EDUCATION_ORDINAL.get(jd_required_education, 0)
    if candidate_edu_ord >= 5 and required_edu_ord <= 3:
        flag = ExpertFlag(
            flag_name="OVERQUALIFICATION_EDU",
            flag_type="PENALTY",
            score_modifier=-0.04,
            dimension_affected="education",
            reason=f"{resume.education_level} for role requiring {jd_required_education}.",
            confidence=0.6,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] OVERQUALIFICATION_EDU: {flag.reason}")

    # ─── LEADERSHIP_MATCH ─────────────────────────────────────
    leadership_kw = [
        "lead", "leader", "manager", "head of", "director", "vp",
        "vice president", "chief", "principal", "team lead", "tech lead",
        "engineering manager", "cto", "ceo", "architect",
    ]
    jd_is_senior = any(
        kw in jd_title.lower() or kw in jd_text_lower
        for kw in ["senior", "lead", "manager", "principal", "head", "director", "chief"]
    )
    candidate_has_leadership = any(
        any(kw in title.lower() for kw in leadership_kw)
        for title in resume.job_titles
    )

    if candidate_has_leadership and jd_is_senior:
        flag = ExpertFlag(
            flag_name="LEADERSHIP_MATCH",
            flag_type="BONUS",
            score_modifier=0.08,
            dimension_affected="miscellaneous",
            reason=f"Leadership experience ({', '.join(resume.job_titles[:2])}) matches senior role.",
            confidence=0.85,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] LEADERSHIP_MATCH: {flag.reason}")
    elif candidate_has_leadership and not jd_is_senior:
        flag = ExpertFlag(
            flag_name="LEADERSHIP_MISMATCH",
            flag_type="INFO",
            score_modifier=-0.02,
            dimension_affected="miscellaneous",
            reason="Leadership background but non-senior role.",
            confidence=0.5,
        )
        flags.append(flag)
        trace_lines.append(f"  [INFO] LEADERSHIP_MISMATCH: {flag.reason}")

    # ─── CAREER_GAP_FLAG ──────────────────────────────────────
    gap_patterns = [
        r"(?i)career\s*(?:break|gap|sabbatical)",
        r"(?i)(?:took|taking)\s+(?:a\s+)?(?:break|time\s+off)",
        r"(?i)gap\s+(?:year|period)",
    ]
    has_gap = any(re.search(p, resume.raw_text) for p in gap_patterns)
    years_mentioned = sorted(set(
        int(y) for y in re.findall(r"\b(20[0-2]\d)\b", resume.raw_text)
    ))
    large_gap = False
    if len(years_mentioned) >= 2:
        for i in range(len(years_mentioned) - 1):
            if years_mentioned[i + 1] - years_mentioned[i] >= 3:
                large_gap = True

    if has_gap or large_gap:
        flag = ExpertFlag(
            flag_name="CAREER_GAP_FLAG",
            flag_type="PENALTY",
            score_modifier=-0.04,
            dimension_affected="miscellaneous",
            reason="Detected career gap (>12 months).",
            confidence=0.6,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] CAREER_GAP_FLAG: {flag.reason}")

    # ─── CAREER_STABILITY_FLAG ────────────────────────────────
    if len(resume.job_titles) >= 4 and resume.experience_years <= 5:
        flag = ExpertFlag(
            flag_name="CAREER_STABILITY_FLAG",
            flag_type="PENALTY",
            score_modifier=-0.05,
            dimension_affected="miscellaneous",
            reason=f"{len(resume.job_titles)} roles in {resume.experience_years:.0f}y. Stability concern.",
            confidence=0.65,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] CAREER_STABILITY_FLAG: {flag.reason}")

    # ─── EXACT_TITLE_MATCH ────────────────────────────────────
    jd_title_lower = jd_title.lower().strip()
    for title in resume.job_titles:
        if jd_title_lower and jd_title_lower in title.lower():
            flag = ExpertFlag(
                flag_name="EXACT_TITLE_MATCH",
                flag_type="BONUS",
                score_modifier=0.07,
                dimension_affected="miscellaneous",
                reason=f"Held exact title '{title}'.",
                confidence=0.9,
            )
            flags.append(flag)
            trace_lines.append(f"  [BONUS] EXACT_TITLE_MATCH: {flag.reason}")
            break

    # ─── FRESH_GRADUATE ───────────────────────────────────────
    if resume.experience_years <= 1:
        flag = ExpertFlag(
            flag_name="FRESH_GRADUATE",
            flag_type="INFO",
            score_modifier=0.0,
            dimension_affected="experience",
            reason="Fresh graduate / early-career profile.",
            confidence=0.75,
        )
        flags.append(flag)
        trace_lines.append(f"  [INFO] FRESH_GRADUATE: {flag.reason}")

    # ─── CERTIFICATION_SCORE ──────────────────────────────────
    cert_keywords = [
        "aws certified", "azure certified", "google certified", "gcp certified",
        "pmp", "scrum master", "safe", "cissp", "cka", "ckad",
        "tensorflow certified", "databricks certified", "snowflake certified",
        "salesforce certified", "istqb", "cfa", "cpa", "acca",
        "six sigma", "itil", "togaf", "prince2",
    ]
    matched_certs = [kw for kw in cert_keywords if kw in resume_text_lower]
    if matched_certs:
        cert_bonus = 0.03 * min(len(matched_certs), 3)
        flag = ExpertFlag(
            flag_name="CERTIFICATION_SCORE",
            flag_type="BONUS",
            score_modifier=cert_bonus,
            dimension_affected="technical_skills",
            reason=f"Certifications: {', '.join(matched_certs[:3])}",
            confidence=0.85,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] CERTIFICATION_SCORE: {flag.reason}")

    # ─── WORK_VISA_FLAG ───────────────────────────────────────
    visa_patterns = [
        r"(?i)visa\s*(?:sponsorship|required)",
        r"(?i)h[-\s]?1b", r"(?i)work\s*permit", r"(?i)employment\s*pass",
    ]
    has_visa_need = any(re.search(p, resume_text_lower) for p in visa_patterns)
    jd_offers_visa = any(re.search(p, jd_text_lower) for p in visa_patterns)

    if has_visa_need and not jd_offers_visa:
        flag = ExpertFlag(
            flag_name="WORK_VISA_FLAG",
            flag_type="PENALTY",
            score_modifier=-0.03,
            dimension_affected="availability",
            reason="May require visa sponsorship.",
            confidence=0.5,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] WORK_VISA_FLAG: {flag.reason}")
    elif jd_offers_visa:
        flag = ExpertFlag(
            flag_name="WORK_VISA_FLAG",
            flag_type="BONUS",
            score_modifier=0.02,
            dimension_affected="availability",
            reason="JD offers visa sponsorship.",
            confidence=0.6,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] WORK_VISA_FLAG: {flag.reason}")

    # ─── RELOCATION_FLAG ──────────────────────────────────────
    reloc_patterns = [
        r"(?i)willing\s+to\s+relocate",
        r"(?i)open\s+to\s+relocation",
        r"(?i)can\s+relocate",
    ]
    if any(re.search(p, resume_text_lower) for p in reloc_patterns):
        flag = ExpertFlag(
            flag_name="RELOCATION_FLAG",
            flag_type="BONUS",
            score_modifier=0.03,
            dimension_affected="availability",
            reason="Willing to relocate.",
            confidence=0.8,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] RELOCATION_FLAG: {flag.reason}")

    # ─── UPSKILL_FLAG ─────────────────────────────────────────
    upskill_patterns = [
        r"(?i)coursera", r"(?i)udemy", r"(?i)edx",
        r"(?i)online\s+course", r"(?i)mooc",
        r"(?i)continuous\s+learning", r"(?i)self[-\s]?taught",
    ]
    upskill_hits = sum(1 for p in upskill_patterns if re.search(p, resume_text_lower))
    if upskill_hits >= 2:
        flag = ExpertFlag(
            flag_name="UPSKILL_FLAG",
            flag_type="BONUS",
            score_modifier=0.03,
            dimension_affected="technical_skills",
            reason="Evidence of continuous learning.",
            confidence=0.7,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] UPSKILL_FLAG: {flag.reason}")

    # ─── EDUCATION_GAP_FLAG ───────────────────────────────────
    if required_edu_ord > 0 and candidate_edu_ord > 0 and candidate_edu_ord < required_edu_ord:
        gap = required_edu_ord - candidate_edu_ord
        flag = ExpertFlag(
            flag_name="EDUCATION_GAP_FLAG",
            flag_type="PENALTY",
            score_modifier=-0.03 * gap,
            dimension_affected="education",
            reason=f"{gap} level(s) below requirement.",
            confidence=0.8,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] EDUCATION_GAP_FLAG: {flag.reason}")

    # ─── Summary ──────────────────────────────────────────────
    bonus_count = sum(1 for f in flags if f.flag_type == "BONUS")
    penalty_count = sum(1 for f in flags if f.flag_type == "PENALTY")
    trace_lines.append(
        f"  SUMMARY: {len(flags)} flags ({bonus_count} bonus, {penalty_count} penalty)"
    )

    return FlagResult(flags=flags, reasoning_trace="\n".join(trace_lines))
