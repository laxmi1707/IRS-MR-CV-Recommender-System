"""
expert_flags.py — Step 2: Expert Flag System (Simplified)
6 Core Flags for Candidate Evaluation

Flags (T/F logic):
1. RELOCATION_FLAG: T=penalize, F=reward
2. WORK_VISA_FLAG: T=penalize, F=reward
3. LEADERSHIP_MATCH: T=reward, F=penalize
4. HIGH_POTENTIAL: T=reward, F=penalize
5. OVERQUALIFICATION_FLAG: T=penalize, F=reward
6. AVAILABILITY_RISK: T=penalize, F=reward
"""

import re
from dataclasses import dataclass, field
from ..resume_processing.resume_parser import ParsedResume, EDUCATION_ORDINAL


@dataclass
class ExpertFlag:
    """Single expert flag with T/F value."""
    flag_name: str
    flag_value: bool  # True/False value
    score_modifier: float  # +/- adjustment
    dimension_affected: str
    reason: str
    confidence: float

    @property
    def flag_type(self) -> str:
        if self.score_modifier > 0:
            return "BONUS"
        if self.score_modifier < 0:
            return "PENALTY"
        return "INFO"


@dataclass
class FlagResult:
    """Result of expert flag assignment."""
    flags: list = field(default_factory=list)
    flag_dict: dict = field(default_factory=dict)  # {flag_name: True/False}
    reasoning_trace: str = ""

    @property
    def total_bonus(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier > 0)

    @property
    def total_penalty(self) -> float:
        return sum(f.score_modifier for f in self.flags if f.score_modifier < 0)

    @property
    def bonus_flags(self) -> list:
        return [f for f in self.flags if f.score_modifier > 0]

    @property
    def penalty_flags(self) -> list:
        return [f for f in self.flags if f.score_modifier < 0]


def assign_expert_flags(
    resume: ParsedResume,
    jd_title: str,
    jd_min_experience: float,
    jd_required_education: str,
    jd_skills: list,
    jd_text: str = "",
) -> FlagResult:
    """
    Assign 6 core expert flags based on candidate profile.
    
    Flag Logic:
    1. RELOCATION_FLAG (T=penalize, F=reward)
       True = candidate needs relocation (constraint)
       False = candidate already local/willing (advantage)
    
    2. WORK_VISA_FLAG (T=penalize, F=reward)
       True = candidate needs visa sponsorship (constraint)
       False = already authorized/no need (advantage)
    
    3. LEADERSHIP_MATCH (T=reward, F=penalize)
       True = candidate has leadership experience for senior role
       False = no leadership match for leadership role
    
    4. HIGH_POTENTIAL (T=reward, F=penalize)
       True = indicators of high growth potential
       False = no growth potential signals
    
    5. OVERQUALIFICATION_FLAG (T=penalize, F=reward)
       True = candidate 2.5x+ overqualified (flight risk)
       False = appropriately qualified
    
    6. AVAILABILITY_RISK (T=penalize, F=reward)
       True = multiple availability constraints
       False = no risk factors
    """
    flags = []
    flag_dict = {}
    trace_lines = [f"=== Expert Flags: {resume.name} ==="]
    
    resume_text_lower = resume.raw_text.lower()
    resume_titles_lower = " ".join(resume.job_titles).lower()
    jd_text_lower = jd_text.lower()
    jd_title_lower = jd_title.lower()
    
    candidate_edu_ord = EDUCATION_ORDINAL.get(resume.education_level, 0)
    
    # ─── 1. RELOCATION_FLAG ──────────────────────────────────
    # True = needs relocation (penalize), False = local/willing (reward)
    relocation_indicators = [
        r"(?i)willing\s+to\s+relocate",
        r"(?i)open\s+to\s+relocation",
        r"(?i)can\s+relocate",
        r"(?i)relocate\s+\w*(?:possible|feasible|available)",
    ]
    willing_to_relocate = any(re.search(p, resume_text_lower) for p in relocation_indicators)
    relocation_flag_value = not willing_to_relocate  # True = NOT willing/needs it = penalize
    
    if relocation_flag_value:  # True: needs relocation
        modifier = -0.05
        flag_type_str = "PENALTY"
        reason = "Requires relocation (not indicated as willing)."
    else:  # False: willing/already local
        modifier = +0.04
        flag_type_str = "BONUS"
        reason = "Open to relocation or already local."
    
    flag = ExpertFlag(
        flag_name="RELOCATION_FLAG",
        flag_value=relocation_flag_value,
        score_modifier=modifier,
        dimension_affected="availability",
        reason=reason,
        confidence=0.75,
    )
    flags.append(flag)
    flag_dict["RELOCATION_FLAG"] = relocation_flag_value
    trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({relocation_flag_value}): {reason}")

    # ─── 2. WORK_VISA_FLAG ───────────────────────────────────
    # True = needs visa (penalize), False = authorized (reward)
    visa_need_patterns = [
        r"(?i)visa\s+(?:sponsorship|required|needed)",
        r"(?i)h[-\s]?1b", r"(?i)work\s+permit", r"(?i)employment\s+pass",
        r"(?i)sponsorship\s+(?:required|needed)",
    ]
    candidate_needs_visa = any(re.search(p, resume_text_lower) for p in visa_need_patterns)
    jd_offers_visa = any(re.search(p, jd_text_lower) for p in visa_need_patterns)
    
    visa_flag_value = candidate_needs_visa and not jd_offers_visa  # True = needs visa but no offer = penalize
    
    if visa_flag_value:  # True: needs visa, no support
        modifier = -0.05
        flag_type_str = "PENALTY"
        reason = "Requires visa sponsorship (not offered by JD)."
    else:  # False: authorized or visa offered
        modifier = +0.03
        flag_type_str = "BONUS"
        reason = "Already authorized or visa sponsorship available."
    
    flag = ExpertFlag(
        flag_name="WORK_VISA_FLAG",
        flag_value=visa_flag_value,
        score_modifier=modifier,
        dimension_affected="availability",
        reason=reason,
        confidence=0.70,
    )
    flags.append(flag)
    flag_dict["WORK_VISA_FLAG"] = visa_flag_value
    trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({visa_flag_value}): {reason}")

    # ─── 3. LEADERSHIP_MATCH ─────────────────────────────────
    # True = has leadership for leadership role (reward), False = no match (penalize)
    leadership_keywords = [
        "lead", "leader", "manager", "head of", "director", "vp",
        "vice president", "chief", "principal", "team lead", "tech lead",
        "engineering manager", "cto", "ceo", "cfo", "coo", "architect",
    ]
    
    jd_is_leadership = any(
        kw in jd_title_lower or kw in jd_text_lower
        for kw in ["senior", "lead", "manager", "principal", "head", "director", "chief", "cto", "ceo"]
    )
    
    candidate_has_leadership = any(
        any(kw in title.lower() for kw in leadership_keywords)
        for title in resume.job_titles
    )
    
    leadership_match_value = jd_is_leadership and candidate_has_leadership  # True = leadership match
    
    if leadership_match_value:  # True: leadership role + leadership experience
        modifier = +0.08
        flag_type_str = "BONUS"
        reason = f"Leadership experience matches senior/leadership role."
    else:  # False: mismatch (either no leadership role or candidate lacks experience)
        if jd_is_leadership and not candidate_has_leadership:
            modifier = -0.05
            flag_type_str = "PENALTY"
            reason = f"Leadership role requires experience candidate lacks."
        else:
            modifier = 0.0
            flag_type_str = "INFO"
            reason = "Non-leadership role; no mismatch."
    
    flag = ExpertFlag(
        flag_name="LEADERSHIP_MATCH",
        flag_value=leadership_match_value,
        score_modifier=modifier,
        dimension_affected="miscellaneous",
        reason=reason,
        confidence=0.80,
    )
    flags.append(flag)
    flag_dict["LEADERSHIP_MATCH"] = leadership_match_value
    if modifier != 0.0:
        trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({leadership_match_value}): {reason}")
    else:
        trace_lines.append(f"  [INFO] {flag.flag_name} ({leadership_match_value}): {reason}")

    # ─── 4. HIGH_POTENTIAL ───────────────────────────────────
    # True = shows growth potential (reward), False = no signals (penalize)
    high_potential_indicators = 0
    
    # Indicator 1: Young with significant experience + advanced education
    if resume.experience_years >= 7 and candidate_edu_ord >= 4:
        high_potential_indicators += 1
    
    # Indicator 2: Multiple professional certifications
    cert_keywords = [
        "aws certified", "azure certified", "gcp certified", "certified",
        "pmp", "scrum master", "cissp", "cka", "istqb", "six sigma",
    ]
    matched_certs = sum(1 for kw in cert_keywords if kw in resume_text_lower)
    if matched_certs >= 2:
        high_potential_indicators += 1
    
    # Indicator 3: Leadership experience + advanced education
    if candidate_has_leadership and candidate_edu_ord >= 4:
        high_potential_indicators += 1
    
    # Indicator 4: Keywords indicating high performer
    potential_keywords = [
        "high performer", "rising star", "top talent", "rapid growth",
        "fast-track", "promoted", "rapid progression", "exceptional",
    ]
    has_potential_signals = any(kw in resume_text_lower for kw in potential_keywords)
    if has_potential_signals:
        high_potential_indicators += 1
    
    high_potential_value = high_potential_indicators >= 2  # True = 2+ indicators
    
    if high_potential_value:  # True: growth indicators present
        modifier = +0.06
        flag_type_str = "BONUS"
        reason = f"High-potential candidate: {high_potential_indicators} growth indicators."
    else:  # False: no indicators
        modifier = -0.02
        flag_type_str = "PENALTY"
        reason = "No significant growth potential indicators detected."
    
    flag = ExpertFlag(
        flag_name="HIGH_POTENTIAL",
        flag_value=high_potential_value,
        score_modifier=modifier,
        dimension_affected="miscellaneous",
        reason=reason,
        confidence=0.70,
    )
    flags.append(flag)
    flag_dict["HIGH_POTENTIAL"] = high_potential_value
    trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({high_potential_value}): {reason}")

    # ─── 5. OVERQUALIFICATION_FLAG ────────────────────────────
    # True = 2.5x+ overqualified (flight risk), False = appropriate level
    overqualification_value = False
    
    if jd_min_experience > 0 and resume.experience_years >= jd_min_experience * 2.5:
        overqualification_value = True
    
    if overqualification_value:  # True: severely overqualified
        modifier = -0.06
        flag_type_str = "PENALTY"
        reason = f"Overqualified: {resume.experience_years:.0f}y vs {jd_min_experience:.0f}y required. Flight risk."
    else:  # False: appropriate level
        modifier = +0.02
        flag_type_str = "BONUS"
        reason = f"Appropriately qualified: {resume.experience_years:.0f}y experience level."
    
    flag = ExpertFlag(
        flag_name="OVERQUALIFICATION_FLAG",
        flag_value=overqualification_value,
        score_modifier=modifier,
        dimension_affected="experience",
        reason=reason,
        confidence=0.75,
    )
    flags.append(flag)
    flag_dict["OVERQUALIFICATION_FLAG"] = overqualification_value
    trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({overqualification_value}): {reason}")

    # ─── 6. AVAILABILITY_RISK ────────────────────────────────
    # True = multiple risk factors (penalize), False = no risk (reward)
    availability_risk_count = 0
    risk_factors = []
    
    # Risk 1: Long notice period (>90 days)
    if resume.notice_period_days > 90:
        availability_risk_count += 1
        risk_factors.append(f"notice>{resume.notice_period_days}d")
    
    # Risk 2: Pursuing advanced degree
    degree_patterns = [
        r"(?i)(?:pursuing|studying)\s+(?:mttech|mtech|mba|ms|phd|llm)",
        r"(?i)(?:mttech|mtech|mba|phd)\s+(?:in\s+progress|ongoing)",
    ]
    pursuing_degree = any(re.search(p, resume_text_lower) for p in degree_patterns)
    if pursuing_degree:
        availability_risk_count += 1
        risk_factors.append("pursuing_advanced_degree")
    
    # Risk 3: Needs visa but no JD offer
    if visa_flag_value:
        availability_risk_count += 1
        risk_factors.append("visa_needs")
    
    # Risk 4: Career instability (many role changes)
    if len(resume.job_titles) >= 5 and resume.experience_years <= 6:
        availability_risk_count += 1
        risk_factors.append("high_job_change_rate")
    
    availability_risk_value = availability_risk_count >= 2  # True = 2+ risk factors
    
    if availability_risk_value:  # True: multiple risks
        modifier = -0.05
        flag_type_str = "PENALTY"
        reason = f"Availability risks: {', '.join(risk_factors)}"
    else:  # False: no major risks
        modifier = +0.03
        flag_type_str = "BONUS"
        reason = "No significant availability risks detected."
    
    flag = ExpertFlag(
        flag_name="AVAILABILITY_RISK",
        flag_value=availability_risk_value,
        score_modifier=modifier,
        dimension_affected="availability",
        reason=reason,
        confidence=0.70,
    )
    flags.append(flag)
    flag_dict["AVAILABILITY_RISK"] = availability_risk_value
    trace_lines.append(f"  [{flag_type_str}] {flag.flag_name} ({availability_risk_value}): {reason}")

    # ─── Summary ──────────────────────────────────────────────
    bonus_count = sum(1 for f in flags if f.score_modifier > 0)
    penalty_count = sum(1 for f in flags if f.score_modifier < 0)
    trace_lines.append(
        f"  SUMMARY: {len(flags)} flags — {bonus_count} bonus, {penalty_count} penalty"
    )

    return FlagResult(
        flags=flags,
        flag_dict=flag_dict,
        reasoning_trace="\n".join(trace_lines)
    )
