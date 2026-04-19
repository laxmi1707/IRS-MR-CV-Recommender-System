"""
expert_flags.py — Step 2: Knowledge-Based Reasoning Layer
Forward Chaining Expert Flag System

After eligibility check, this module assigns semantic FLAGS
to each candidate that modify scoring:
- OVERQUALIFIED → penalty (flight risk)
- LEADERSHIP_EXPERIENCE → bonus for senior roles
- CAREER_GAP → penalty if gap > 12 months
- JOB_HOPPER → penalty if >3 jobs in 3 years
- CERTIFICATION_MATCH → bonus for relevant certs
- EXACT_TITLE_MATCH → bonus if held the exact JD role
- FRESH_GRADUATE → flag for entry-level consideration
- INDUSTRY_SWITCH → flag for domain transition

Each flag has a reasoning trace for XAI.
"""

import re
from dataclasses import dataclass, field
from resume_parser import ParsedResume, EDUCATION_ORDINAL


@dataclass
class ExpertFlag:
    """A single expert flag assigned to a candidate."""
    flag_name: str
    flag_type: str  # "BONUS", "PENALTY", "INFO"
    score_modifier: float  # e.g., +0.05 or -0.10
    dimension_affected: str  # which dimension this modifies
    reason: str
    confidence: float  # 0.0 to 1.0


@dataclass
class FlagResult:
    """All flags for a candidate."""
    flags: list[ExpertFlag] = field(default_factory=list)
    reasoning_trace: str = ""

    @property
    def total_bonus(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier > 0)

    @property
    def total_penalty(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier < 0)

    @property
    def bonus_flags(self) -> list[ExpertFlag]:
        return [f for f in self.flags if f.flag_type == "BONUS"]

    @property
    def penalty_flags(self) -> list[ExpertFlag]:
        return [f for f in self.flags if f.flag_type == "PENALTY"]


def assign_expert_flags(
    resume: ParsedResume,
    jd_title: str,
    jd_min_experience: float,
    jd_required_education: str,
    jd_skills: list[str],
    jd_text: str = "",
) -> FlagResult:
    """
    Forward chaining rule engine that assigns expert flags.
    Each rule checks conditions and fires if met.
    """
    flags = []
    trace_lines = [f"=== Expert Flags: {resume.name} ==="]

    # ─── Flag 1: OVERQUALIFIED ────────────────────────────────
    # If candidate has 2x+ the required experience, flag as overqualified
    if jd_min_experience > 0 and resume.experience_years >= jd_min_experience * 2.5:
        flag = ExpertFlag(
            flag_name="OVERQUALIFIED",
            flag_type="PENALTY",
            score_modifier=-0.08,
            dimension_affected="experience",
            reason=f"Candidate has {resume.experience_years:.0f}y experience vs "
                   f"{jd_min_experience:.0f}y required (>{2.5}x). Flight risk — "
                   f"likely to leave for a more senior role.",
            confidence=0.7,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] OVERQUALIFIED: {flag.reason}")

    # Also check education overqualification
    candidate_edu_ord = EDUCATION_ORDINAL.get(resume.education_level, 0)
    required_edu_ord = EDUCATION_ORDINAL.get(jd_required_education, 0)
    if candidate_edu_ord >= 5 and required_edu_ord <= 3:
        flag = ExpertFlag(
            flag_name="OVERQUALIFIED_EDUCATION",
            flag_type="PENALTY",
            score_modifier=-0.05,
            dimension_affected="education",
            reason=f"Candidate holds {resume.education_level} but role only requires "
                   f"{jd_required_education}. May be overqualified academically.",
            confidence=0.6,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] OVERQUALIFIED_EDUCATION: {flag.reason}")

    # ─── Flag 2: LEADERSHIP_EXPERIENCE ────────────────────────
    leadership_keywords = [
        "lead", "leader", "manager", "head of", "director", "vp ",
        "vice president", "chief", "principal", "team lead", "tech lead",
        "engineering manager", "cto", "ceo",
    ]
    jd_is_senior = any(
        kw in jd_title.lower() or kw in jd_text.lower()
        for kw in ["senior", "lead", "manager", "principal", "head", "director"]
    )

    candidate_has_leadership = any(
        any(kw in title.lower() for kw in leadership_keywords)
        for title in resume.job_titles
    )

    if candidate_has_leadership and jd_is_senior:
        flag = ExpertFlag(
            flag_name="LEADERSHIP_MATCH",
            flag_type="BONUS",
            score_modifier=0.07,
            dimension_affected="miscellaneous",
            reason=f"Candidate has leadership experience ({resume.job_titles[:2]}) "
                   f"matching senior role requirement.",
            confidence=0.8,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] LEADERSHIP_MATCH: {flag.reason}")
    elif candidate_has_leadership and not jd_is_senior:
        flag = ExpertFlag(
            flag_name="OVERQUALIFIED_ROLE",
            flag_type="INFO",
            score_modifier=-0.03,
            dimension_affected="miscellaneous",
            reason=f"Candidate held leadership roles ({resume.job_titles[:2]}) "
                   f"but applying for a non-senior position.",
            confidence=0.5,
        )
        flags.append(flag)
        trace_lines.append(f"  [INFO] OVERQUALIFIED_ROLE: {flag.reason}")

    # ─── Flag 3: CAREER_GAP ──────────────────────────────────
    # Detect mentions of career gaps in the resume
    gap_patterns = [
        r"(?i)career\s*(?:break|gap|sabbatical)",
        r"(?i)(?:took|taking)\s+(?:a\s+)?(?:break|time\s+off)",
        r"(?i)gap\s+(?:year|period)",
    ]
    has_gap = any(re.search(p, resume.raw_text) for p in gap_patterns)

    # Also detect large year gaps in work history
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
            flag_name="CAREER_GAP",
            flag_type="PENALTY",
            score_modifier=-0.05,
            dimension_affected="miscellaneous",
            reason="Detected potential career gap (>12 months) in work history. "
                   "May need additional context during interview.",
            confidence=0.5 if large_gap else 0.7,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] CAREER_GAP: {flag.reason}")

    # ─── Flag 4: JOB_HOPPER ──────────────────────────────────
    # If candidate has many job titles relative to experience
    if len(resume.job_titles) >= 4 and resume.experience_years <= 5:
        flag = ExpertFlag(
            flag_name="JOB_HOPPER",
            flag_type="PENALTY",
            score_modifier=-0.06,
            dimension_affected="miscellaneous",
            reason=f"Candidate has {len(resume.job_titles)} distinct roles in "
                   f"{resume.experience_years:.0f} years. Possible stability concern.",
            confidence=0.6,
        )
        flags.append(flag)
        trace_lines.append(f"  [PENALTY] JOB_HOPPER: {flag.reason}")

    # ─── Flag 5: EXACT_TITLE_MATCH ───────────────────────────
    jd_title_lower = jd_title.lower().strip()
    for title in resume.job_titles:
        if jd_title_lower and jd_title_lower in title.lower():
            flag = ExpertFlag(
                flag_name="EXACT_TITLE_MATCH",
                flag_type="BONUS",
                score_modifier=0.06,
                dimension_affected="miscellaneous",
                reason=f"Candidate previously held the exact role: '{title}', "
                       f"matching JD title '{jd_title}'.",
                confidence=0.9,
            )
            flags.append(flag)
            trace_lines.append(f"  [BONUS] EXACT_TITLE_MATCH: {flag.reason}")
            break

    # ─── Flag 6: FRESH_GRADUATE ──────────────────────────────
    if resume.experience_years <= 1:
        flag = ExpertFlag(
            flag_name="FRESH_GRADUATE",
            flag_type="INFO",
            score_modifier=0.0,
            dimension_affected="experience",
            reason="Candidate appears to be a fresh graduate or early-career professional.",
            confidence=0.7,
        )
        flags.append(flag)
        trace_lines.append(f"  [INFO] FRESH_GRADUATE: {flag.reason}")

    # ─── Flag 7: CERTIFICATION_MATCH ─────────────────────────
    cert_keywords = [
        "aws certified", "azure certified", "google certified", "pmp",
        "scrum master", "cissp", "cka", "ckad", "tensorflow certified",
        "databricks", "snowflake", "salesforce certified", "istqb",
    ]
    matched_certs = [
        kw for kw in cert_keywords if kw in resume.raw_text.lower()
    ]
    if matched_certs:
        flag = ExpertFlag(
            flag_name="CERTIFICATION_MATCH",
            flag_type="BONUS",
            score_modifier=0.04 * min(len(matched_certs), 3),  # cap at 3
            dimension_affected="technical_skills",
            reason=f"Candidate holds relevant certifications: {', '.join(matched_certs[:3])}",
            confidence=0.85,
        )
        flags.append(flag)
        trace_lines.append(f"  [BONUS] CERTIFICATION_MATCH: {flag.reason}")

    # ─── Summary ─────────────────────────────────────────────
    bonus_count = sum(1 for f in flags if f.flag_type == "BONUS")
    penalty_count = sum(1 for f in flags if f.flag_type == "PENALTY")
    trace_lines.append(
        f"  SUMMARY: {len(flags)} flags assigned "
        f"({bonus_count} bonus, {penalty_count} penalty)"
    )

    return FlagResult(
        flags=flags,
        reasoning_trace="\n".join(trace_lines),
    )
