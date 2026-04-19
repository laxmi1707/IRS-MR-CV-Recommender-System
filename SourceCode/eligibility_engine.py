"""
eligibility_engine.py — Step 1: Decision Automation Layer
Forward Chaining Inference Engine for Candidate Eligibility

Applies deterministic IF-THEN business rules to filter out
Non-Applicable (NA) candidates BEFORE scoring.
E.g., a Yoga Teacher resume for a Mainframe Dev JD → NA

Each rule generates a reasoning trace for XAI transparency.
"""

import re
from dataclasses import dataclass, field
from resume_parser import ParsedResume


@dataclass
class RuleFiring:
    """Record of a single rule firing in the inference chain."""
    rule_name: str
    condition: str
    result: str  # "PASS", "FAIL", "WARNING"
    details: str


@dataclass
class EligibilityResult:
    """Result of eligibility check for one candidate."""
    is_eligible: bool
    reason: str  # "ELIGIBLE" or reason for rejection
    rules_fired: list[RuleFiring] = field(default_factory=list)
    skill_match_ratio: float = 0.0
    reasoning_trace: str = ""


# ─── Domain Mismatch Detection ────────────────────────────────
# Maps job domain keywords to incompatible resume domains
DOMAIN_KEYWORDS = {
    "software": ["chef", "cook", "culinary", "yoga", "fitness trainer",
                 "beautician", "salon", "hairdresser", "plumber",
                 "carpenter", "electrician", "mechanic", "driver",
                 "gardener", "florist", "tailor", "fashion design"],
    "data_science": ["chef", "cook", "culinary", "yoga", "fitness",
                     "beautician", "salon", "plumber", "carpenter",
                     "electrician", "mechanic", "driver", "gardener",
                     "nursing", "physiotherapy"],
    "finance": ["chef", "cook", "yoga", "fitness", "beautician",
                "plumber", "carpenter", "electrician", "mechanic",
                "gardener", "fashion design", "game design"],
    "engineering": ["chef", "cook", "yoga", "fitness", "beautician",
                    "salon", "florist", "tailor", "fashion design"],
}

# Job title patterns that signal clearly wrong domain
INCOMPATIBLE_TITLE_PATTERNS = [
    (r"(?i)\b(software|data|ml|ai|cloud|devops)\b",
     r"(?i)\b(chef|yoga|beautician|salon|fitness\s*trainer|plumber|carpenter)\b"),
    (r"(?i)\b(finance|banking|accounting)\b",
     r"(?i)\b(chef|yoga|beautician|game\s*designer|fashion)\b"),
]


def _detect_jd_domain(jd_text: str) -> str:
    """Infer the domain of the JD."""
    jd_lower = jd_text.lower()
    domain_scores = {}
    for domain, _ in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in [
            "software", "python", "java", "api", "database", "cloud",
            "react", "backend", "frontend", "devops", "kubernetes",
        ] if kw in jd_lower) if domain == "software" else 0

        score += sum(1 for kw in [
            "data", "machine learning", "statistics", "analytics",
            "model", "tensorflow", "pandas", "sql",
        ] if kw in jd_lower) if domain == "data_science" else 0

        score += sum(1 for kw in [
            "finance", "accounting", "banking", "audit", "compliance",
            "risk", "investment", "portfolio",
        ] if kw in jd_lower) if domain == "finance" else 0

        score += sum(1 for kw in [
            "engineer", "mechanical", "civil", "electrical", "hardware",
            "circuit", "cad", "manufacturing",
        ] if kw in jd_lower) if domain == "engineering" else 0

        domain_scores[domain] = score

    best_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[best_domain] == 0:
        return "general"
    return best_domain


def check_eligibility(
    resume: ParsedResume,
    jd_required_skills: list[str],
    jd_min_experience: float,
    jd_min_education: str,
    jd_text: str,
    jd_title: str = "",
    sbert_model=None,
) -> EligibilityResult:
    """
    Forward chaining eligibility check.

    Rules are fired sequentially. If any HARD rule fails,
    the candidate is marked NA (Not Applicable).

    Rule order:
    1. Domain Mismatch Rule (hardest filter)
    2. Minimum Skills Overlap Rule
    3. Experience Floor Rule
    4. Education Floor Rule (only if JD explicitly requires)

    Returns EligibilityResult with full reasoning trace.
    """
    rules_fired = []
    is_eligible = True
    fail_reason = ""

    resume_text_lower = resume.raw_text.lower()
    resume_titles = [t.lower() for t in resume.job_titles]
    resume_skills_lower = set(s.lower() for s in resume.skills)
    jd_skills_lower = set(s.lower() for s in jd_required_skills)

    # ─── Rule 1: Domain Mismatch ──────────────────────────────
    jd_domain = _detect_jd_domain(jd_text)
    incompatible_keywords = DOMAIN_KEYWORDS.get(jd_domain, [])

    domain_mismatch_count = 0
    matched_incompatible = []
    for kw in incompatible_keywords:
        if kw in resume_text_lower:
            domain_mismatch_count += 1
            matched_incompatible.append(kw)

    # Also check if resume job titles are completely unrelated
    title_mismatch = False
    for jd_pattern, resume_pattern in INCOMPATIBLE_TITLE_PATTERNS:
        if re.search(jd_pattern, jd_text) and any(
            re.search(resume_pattern, t) for t in resume_titles
        ):
            title_mismatch = True

    if domain_mismatch_count >= 2 or title_mismatch:
        rule = RuleFiring(
            rule_name="DomainMismatchRule",
            condition=f"JD domain='{jd_domain}', resume contains incompatible terms: {matched_incompatible[:5]}",
            result="FAIL",
            details=f"Resume appears to be from a different professional domain. "
                    f"Found {domain_mismatch_count} incompatible domain keywords."
        )
        rules_fired.append(rule)
        is_eligible = False
        fail_reason = f"Domain mismatch: resume domain incompatible with {jd_domain} role"
    else:
        rules_fired.append(RuleFiring(
            rule_name="DomainMismatchRule",
            condition=f"JD domain='{jd_domain}'",
            result="PASS",
            details="No significant domain mismatch detected."
        ))

    # ─── Rule 2: Minimum Skills Overlap ───────────────────────
    if is_eligible and jd_skills_lower:
        # Use basic keyword matching for eligibility (SBERT is for scoring)
        matched_skills = resume_skills_lower & jd_skills_lower
        skill_ratio = len(matched_skills) / len(jd_skills_lower) if jd_skills_lower else 0

        # Also do semantic check if SBERT available
        semantic_matches = 0
        if sbert_model and jd_skills_lower - matched_skills:
            import numpy as np
            unmatched_jd = list(jd_skills_lower - matched_skills)
            resume_skill_list = list(resume_skills_lower)
            if resume_skill_list and unmatched_jd:
                jd_embs = sbert_model.encode(unmatched_jd)
                res_embs = sbert_model.encode(resume_skill_list)
                for j, jd_emb in enumerate(jd_embs):
                    sims = [
                        float(np.dot(jd_emb, r) / (np.linalg.norm(jd_emb) * np.linalg.norm(r) + 1e-8))
                        for r in res_embs
                    ]
                    if max(sims) >= 0.75:
                        semantic_matches += 1
                        matched_skills.add(f"~{unmatched_jd[j]}")

            total_matched = len(matched_skills)
            skill_ratio = total_matched / len(jd_skills_lower)

        # NA ONLY if absolutely ZERO skills match (no rejection policy)
        # A Java developer applying for Data Analyst still gets scored (low)
        # Only Yoga Teacher / Plumber with 0/12 skills → NA
        if skill_ratio <= 0.0 and len(matched_skills) == 0 and semantic_matches == 0:
            rule = RuleFiring(
                rule_name="MinSkillOverlapRule",
                condition=f"Skill overlap = 0/{len(jd_skills_lower)} (zero match)",
                result="FAIL",
                details=f"Zero technical skills matched out of {len(jd_skills_lower)} required. "
                        f"Candidate's profile is from a completely unrelated domain."
            )
            rules_fired.append(rule)
            is_eligible = False
            fail_reason = f"Zero skill match (0/{len(jd_skills_lower)} — unrelated domain)"
        else:
            rules_fired.append(RuleFiring(
                rule_name="MinSkillOverlapRule",
                condition=f"Skill overlap ratio = {skill_ratio:.0%}",
                result="PASS",
                details=f"Matched {len(matched_skills)} of {len(jd_skills_lower)} required skills."
            ))
    else:
        skill_ratio = 0.5  # neutral if no JD skills specified


    # ─── Rule 3: Experience Floor ─────────────────────────────
    if is_eligible and jd_min_experience > 0:
        rules_fired.append(RuleFiring(
            rule_name="ExperienceFloorRule",
            condition=f"Candidate exp={resume.experience_years}y, required={jd_min_experience}y",
            result="PASS",
            details="No rejection policy — experience gap handled via scoring penalty in D2."
        ))

    # ─── Rule 4: Education Floor ──────────────────────────────
    from resume_parser import EDUCATION_ORDINAL
    if is_eligible and jd_min_education:
        candidate_edu_ord = EDUCATION_ORDINAL.get(resume.education_level, 0)
        required_edu_ord = EDUCATION_ORDINAL.get(jd_min_education, 0)

        # Check for specialized degree mismatch (e.g., LLB vs BE)
        # These are in different streams — higher degree in wrong stream doesn't count
        SPECIALIZED_DEGREES = {
            "llb": "law",
            "law": "law",
            "llm": "law",
            "mbbs": "medicine",
            "md": "medicine",
            "bds": "medicine",
            "medicine": "medicine",
            "b.arch": "architecture",
            "architecture": "architecture",
            "bfa": "arts",
            "fine arts": "arts",
            "b.ed": "education",
            "education": "education",
        }

        # Detect if JD requires a specialized degree
        jd_edu_lower = jd_min_education.lower()
        jd_text_lower = jd_text.lower()
        jd_stream = None
        for keyword, stream in SPECIALIZED_DEGREES.items():
            if keyword in jd_edu_lower or keyword in jd_text_lower:
                jd_stream = stream
                break

        # Detect candidate's stream
        candidate_stream = None
        resume_edu_lower = (resume.education_text + " " + resume.education_level).lower()
        for keyword, stream in SPECIALIZED_DEGREES.items():
            if keyword in resume_edu_lower:
                candidate_stream = stream
                break

        # Rule logic:
        # Case 1: JD requires specialized degree (LLB, MBBS etc.)
        #         and candidate is from a DIFFERENT stream → NA
        if jd_stream and candidate_stream != jd_stream:
            rule = RuleFiring(
                rule_name="EducationFloorRule",
                condition=f"JD requires '{jd_stream}' stream degree, "
                          f"candidate has '{candidate_stream or 'general'}' stream",
                result="FAIL",
                details=f"JD requires a specialized {jd_stream} degree. "
                        f"Candidate's education is in a different stream. "
                        f"A higher degree in a different field does not qualify.",
            )
            rules_fired.append(rule)
            is_eligible = False
            fail_reason = f"Wrong education stream ({candidate_stream or 'general'} vs required {jd_stream})"

        # Case 2: JD requires general degree (Bachelors/Masters etc.)
        #         Candidate has LOWER level → NA
        #         e.g., JD requires BE and candidate has Diploma → NA
        #         But candidate with MTech (higher than BE) → PASS
        elif required_edu_ord > 0 and candidate_edu_ord < required_edu_ord:
            rule = RuleFiring(
                rule_name="EducationFloorRule",
                condition=f"Candidate: '{resume.education_level}' (level {candidate_edu_ord}), "
                          f"Required: '{jd_min_education}' (level {required_edu_ord})",
                result="FAIL",
                details=f"JD requires minimum {jd_min_education} (level {required_edu_ord}). "
                        f"Candidate has {resume.education_level} (level {candidate_edu_ord}). "
                        f"Education level is below minimum requirement.",
            )
            rules_fired.append(rule)
            is_eligible = False
            fail_reason = (f"Education below requirement "
                          f"({resume.education_level} < {jd_min_education})")

        # Case 3: Candidate meets or exceeds → PASS
        else:
            rules_fired.append(RuleFiring(
                rule_name="EducationFloorRule",
                condition=f"Candidate: '{resume.education_level}' (level {candidate_edu_ord}), "
                          f"Required: '{jd_min_education}' (level {required_edu_ord})",
                result="PASS",
                details=f"Education meets or exceeds requirement. "
                        f"{'Higher degree — eligible.' if candidate_edu_ord > required_edu_ord else 'Matches requirement.'}",
            ))

            
    # ─── Build Reasoning Trace ────────────────────────────────
    trace_lines = [f"=== Eligibility Check: {resume.name} ==="]
    for rf in rules_fired:
        trace_lines.append(
            f"  [{rf.result}] {rf.rule_name}: {rf.condition}\n"
            f"         → {rf.details}"
        )
    if is_eligible:
        trace_lines.append("  VERDICT: ELIGIBLE — candidate proceeds to scoring.")
    else:
        trace_lines.append(f"  VERDICT: NOT APPLICABLE — {fail_reason}")

    return EligibilityResult(
        is_eligible=is_eligible,
        reason="ELIGIBLE" if is_eligible else fail_reason,
        rules_fired=rules_fired,
        skill_match_ratio=skill_ratio,
        reasoning_trace="\n".join(trace_lines),
    )
