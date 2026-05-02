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
from ..resume_processing.resume_parser import ParsedResume


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
    # Fitness / wellness
    "yoga instructor", "yoga teacher", "fitness trainer", "personal trainer",
    "pilates instructor", "pilates trainer", "pilates teacher",
    "group fitness instructor", "aerobics instructor",
    # Culinary
    "chef", "sous chef", "head chef", "pastry chef", "executive chef", "cook",
    # Beauty / personal care
    "hairdresser", "barber", "beautician", "cosmetologist", "makeup artist",
    "nail technician", "esthetician",
    # Trades
    "plumber", "electrician", "carpenter", "welder", "mason", "painter",
    "roofer", "locksmith", "machinist",
    # Outdoor / agriculture
    "gardener", "florist", "landscaper", "farmer",
    # Driving
    "truck driver", "cab driver", "delivery driver", "taxi driver", "bus driver",
    # Service
    "security guard", "bouncer", "doorman",
    "waiter", "waitress", "bartender", "barista", "cashier",
    # Healthcare (separate field)
    "nurse", "midwife", "paramedic", "caregiver", "physical therapist",
    # Other
    "teacher", "tutor",  # academic teaching is its own field
]

# Domain categories for profession relevance checking
PROFESSIONAL_DOMAINS = {
    "tech": ["software", "developer", "engineer", "programmer", "devops", "qa", "testing", "analyst"],
    "finance": ["finance", "accounting", "accountant", "auditor", "financial", "bank", "banking", "treasury", "investment"],
    "management": ["manager", "director", "lead", "coordinator", "supervisor", "head", "executive"],
    "sales": ["sales", "business development", "account executive", "representative"],
    "hr": ["hr", "human resources", "recruiter", "recruitment", "talent"],
    "marketing": ["marketing", "brand", "product", "growth", "digital"],
    "operations": ["operations", "supply chain", "logistics", "production"],
    "uat": ["uat", "testing", "quality", "qa", "test manager"],
}


def _kw_match(keyword: str, text: str) -> bool:
    """Word-boundary match — avoids 'cook' matching 'cookies'."""
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


# Tech-context keywords that signal a tech resume.
# If 5+ appear, we raise the threshold for unrelated-profession matches in body text.
TECH_CONTEXT_HINTS = [
    "software", "engineer", "developer", "java", "python", "javascript",
    "kubernetes", "docker", "aws", "azure", "api", "microservice", "devops",
    "ci/cd", "agile", "scrum", "git", "sql", "machine learning", "data science",
    "architect", "framework", "deployment", "cloud", "backend", "frontend",
    "selenium", "cucumber", "automation", "qa", "test",
]


def _semantic_profession_match(resume: ParsedResume, jd_title: str, jd_text: str,
                                sbert_model) -> tuple[bool, float, str]:
    """Use SBERT to compare candidate's profession with JD's profession.

    Returns (is_aligned, similarity, explanation).

    This is the robust way to detect unrelated professions — works even if
    the candidate's job title isn't in our keyword list (e.g. 'painter' for
    a software engineer JD). We compute a similarity between:
      - candidate's job_titles (joined) and the JD title + first sentence
      - if similarity < 0.30 AND no skill overlap, candidate is unrelated
    """
    if sbert_model is None:
        return True, 1.0, "SBERT unavailable, skipped semantic check"

    cand_profile = " ".join(resume.job_titles[:5]) if resume.job_titles else ""
    if not cand_profile:
        # Fall back to first 200 chars of resume — usually contains role context
        cand_profile = resume.raw_text[:200]

    # JD profile: title + first sentence of description for context
    jd_first = jd_text.split(".")[0] if jd_text else ""
    jd_profile = (jd_title or "") + ". " + jd_first

    if not jd_profile.strip() or not cand_profile.strip():
        return True, 1.0, "Insufficient data for semantic check"

    try:
        import numpy as np
        cand_emb = sbert_model.encode(cand_profile)
        jd_emb = sbert_model.encode(jd_profile)
        sim = float(np.dot(cand_emb, jd_emb) / (
            np.linalg.norm(cand_emb) * np.linalg.norm(jd_emb) + 1e-8))
        return sim >= 0.30, sim, f"Profession similarity={sim:.2f}"
    except Exception as e:
        return True, 1.0, f"Semantic check error: {e}"


def _check_unrelated_profession(resume: ParsedResume, jd_title: str, jd_text: str,
                                 jd_skills: list, sbert_model=None):
    """Determine if candidate's profession is fundamentally unrelated to the JD.

    Three-pass approach:
      1. KEYWORD pass — if candidate's job_titles explicitly contain a known
         non-tech keyword (yoga, chef, plumber, painter, etc.) AND the JD is
         NOT for that domain, mark NA. Strongest, most reliable signal.
      2. SBERT SEMANTIC pass — compare candidate's profession to JD's profession
         via cosine similarity. If similarity is very low (< 0.30) AND candidate
         has zero skill overlap with the JD, mark NA. This catches unseen
         professions (painter, cobbler, etc.) without requiring a keyword list.
      3. JD-SIDE keyword pass — if JD is for an unrelated profession (e.g.
         'Yoga Reformer Trainer' or 'Chef'), and the candidate's titles do NOT
         contain that profession's keyword, mark NA.

    Returns (is_unrelated: bool, matched_keywords: list, reason: str).
    """
    resume_text_lower = resume.raw_text.lower()
    resume_titles_lower = " ".join(resume.job_titles).lower()
    jd_lower = (jd_title + " " + jd_text).lower()

    # ─── Pass 1: Direct keyword match in titles ─────────────
    # Check if candidate's titles have a non-tech keyword
    cand_unrelated_kws = [kw for kw in UNRELATED_PROFESSION_KEYWORDS
                          if _kw_match(kw, resume_titles_lower)]

    # Check if JD is itself for one of these domains
    jd_unrelated_kws = [kw for kw in UNRELATED_PROFESSION_KEYWORDS
                        if _kw_match(kw, jd_lower)]

    if cand_unrelated_kws:
        # Candidate is from a non-tech profession.
        # Define profession families (single-word anchors that appear in JDs).
        FAMILY_ANCHORS = [
            {"yoga", "fitness", "pilates", "personal trainer", "trainer", "instructor", "aerobics", "zumba"},  # fitness
            {"chef", "cook", "culinary", "kitchen", "baker", "pastry", "sous"},  # culinary
            {"hairdresser", "beautician", "cosmetologist", "barber", "salon", "spa"},  # beauty
            {"plumber", "electrician", "carpenter", "welder", "mason", "painter", "construction"},  # trades
            {"truck driver", "cab driver", "delivery driver", "driver", "chauffeur"},  # driving
            {"waiter", "waitress", "bartender", "barista", "server"},  # restaurant service
            {"nurse", "midwife", "paramedic", "caregiver", "physical therapist"},  # healthcare
        ]

        cand_blob = resume_titles_lower + " " + resume_text_lower[:500]
        jd_blob = jd_lower

        # Find which family the candidate is in
        cand_fam_idx = None
        for i, fam in enumerate(FAMILY_ANCHORS):
            if any(_kw_match(kw, cand_blob) for kw in fam):
                cand_fam_idx = i
                break

        # Find which family the JD targets
        jd_fam_idx = None
        for i, fam in enumerate(FAMILY_ANCHORS):
            if any(_kw_match(kw, jd_blob) for kw in fam):
                jd_fam_idx = i
                break

        if cand_fam_idx is not None and cand_fam_idx == jd_fam_idx:
            # Both in same family → eligible (yoga teacher → pilates trainer)
            return False, [], "Candidate and JD share profession family"
        elif jd_fam_idx is not None:
            # JD is for a specialty profession but candidate is in a different family → NA
            return True, cand_unrelated_kws, \
                f"Candidate is from a different profession family than the JD ({cand_unrelated_kws[0]})"
        else:
            # JD is generic/tech, candidate is yoga teacher → NA
            return True, cand_unrelated_kws, \
                f"Candidate is from unrelated profession ({cand_unrelated_kws[0]})"

    # ─── Pass 2: JD-side check ──────────────────────────────
    # If JD is for a specialty profession (e.g. Pilates Trainer) but candidate
    # has no matching keyword in titles, they're NA.
    if jd_unrelated_kws:
        # Tech engineer applying for chef job — should be NA
        return True, [], \
            f"JD requires specialty profession ({jd_unrelated_kws[0]}); candidate's profile does not match"

    # ─── Pass 3: SBERT semantic profession check ────────────
    # Catches unseen professions not in our keyword list (e.g. 'cobbler').
    aligned, sim, sem_msg = _semantic_profession_match(resume, jd_title, jd_text, sbert_model)
    if not aligned:
        # Verify no skill overlap before declaring NA — semantic alone is risky
        resume_skills_set = set(s.lower() for s in resume.skills)
        jd_skills_set = set(s.lower() for s in jd_skills)
        skill_overlap = len(resume_skills_set & jd_skills_set)
        if skill_overlap == 0:
            return True, [], f"Low semantic similarity ({sim:.2f}) and zero skill overlap"

    # ─── Body-text fallback (existing logic, preserved) ─────
    # Even if titles weren't extracted, scan body for unrelated-profession keywords
    tech_hits = sum(1 for hint in TECH_CONTEXT_HINTS if _kw_match(hint, resume_text_lower))
    is_tech_resume = tech_hits >= 5
    body_matches = [kw for kw in UNRELATED_PROFESSION_KEYWORDS
                    if _kw_match(kw, resume_text_lower)]
    threshold = 5 if is_tech_resume else 3
    if len(body_matches) >= threshold:
        return True, body_matches, f"Body text contains {len(body_matches)} unrelated-profession keywords"

    return False, [], "Profession appears compatible with JD"


def check_eligibility(
    resume: ParsedResume,
    jd_required_skills: list,
    jd_min_experience: float,
    jd_min_education: str,
    jd_text: str,
    jd_title: str = "",
    sbert_model=None,
    jd_max_experience: float = 0.0,
) -> EligibilityResult:
    """Relaxed eligibility check — defaults to ELIGIBLE."""
    rules_fired = []
    is_eligible = True
    fail_reason = ""
    skill_ratio = 0.5

    resume_skills_lower = set(s.lower() for s in resume.skills)
    jd_skills_lower = set(s.lower() for s in jd_required_skills)

    # ─── Rule 1: Unrelated Profession ─────────────────────────
    is_unrelated, matched_kw, prof_reason = _check_unrelated_profession(
        resume, jd_title, jd_text, jd_required_skills, sbert_model=sbert_model
    )
    if is_unrelated:
        kw_str = ', '.join(matched_kw[:3]) if matched_kw else prof_reason
        rules_fired.append(RuleFiring(
            rule_name="UnrelatedProfessionRule",
            condition=prof_reason,
            result="FAIL",
            details=f"Resume from unrelated profession. {prof_reason}",
        ))
        is_eligible = False
        if matched_kw:
            fail_reason = (
                f"Profession mismatch: the candidate's background ({kw_str}) "
                f"is not related to this role."
            )
        else:
            fail_reason = f"Profession mismatch: {prof_reason}"
    else:
        rules_fired.append(RuleFiring(
            rule_name="UnrelatedProfessionRule",
            condition=prof_reason,
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
            fail_reason = (
                f"Zero skill match: none of the {len(jd_skills_lower)} required skills "
                f"are present in the resume (the candidate has {len(resume.skills)} listed skills, "
                f"none overlapping)."
            )
        else:
            rules_fired.append(RuleFiring(
                rule_name="SkillOverlapRule",
                condition=f"{total_matches}/{len(jd_skills_lower)} matched ({skill_ratio:.0%})",
                result="PASS",
                details="Will be scored on D1.",
            ))

    # ─── Rule 3: Experience window with 10% tolerance ─────────
    # If JD specifies a RANGE (e.g. "4-11 years"), candidate must fit
    # within [min - 10%, max + 10%]. Otherwise (single threshold), the
    # candidate just needs to clear the floor.
    # Examples:
    #   JD "10y" → tolerance 1y → 9y OK, 8y NA
    #   JD "4-11y" → window [4-1, 11+1] = [3, 12] → 3y OK, 12y OK, 13y NA, 2y NA
    if is_eligible and jd_min_experience > 0:
        import math
        cand_yrs = resume.experience_years
        # 10% tolerance computed against the bound being checked
        lower_tolerance = max(1.0, math.floor(jd_min_experience * 0.10))
        floor = jd_min_experience - lower_tolerance

        # Range-mode if a max was provided AND it's strictly greater than min
        is_range_mode = jd_max_experience > jd_min_experience
        if is_range_mode:
            upper_tolerance = max(1.0, math.floor(jd_max_experience * 0.10))
            ceiling = jd_max_experience + upper_tolerance
        else:
            ceiling = float("inf")

        below_floor = cand_yrs < floor
        above_ceiling = is_range_mode and cand_yrs > ceiling

        if below_floor or above_ceiling:
            if below_floor:
                cond = (f"exp={cand_yrs}y, required={jd_min_experience:.0f}+ "
                        f"(floor={floor:.0f}y)")
                if is_range_mode:
                    fail_reason = (
                        f"Outside experience window: candidate has {cand_yrs:.0f}y; "
                        f"the role asks for {jd_min_experience:.0f}–{jd_max_experience:.0f}y "
                        f"(acceptable range with 10% tolerance: {floor:.0f}–{ceiling:.0f}y)."
                    )
                else:
                    fail_reason = (
                        f"Insufficient experience: candidate has {cand_yrs:.0f}y "
                        f"but the role requires {jd_min_experience:.0f}y "
                        f"(minimum acceptable: {floor:.0f}y)."
                    )
            else:
                # above_ceiling — only possible in range mode
                cond = (f"exp={cand_yrs}y, range={jd_min_experience:.0f}-{jd_max_experience:.0f}y "
                        f"(ceiling={ceiling:.0f}y)")
                fail_reason = (
                    f"Above experience window: candidate has {cand_yrs:.0f}y; "
                    f"the role asks for {jd_min_experience:.0f}–{jd_max_experience:.0f}y "
                    f"(acceptable range with 10% tolerance: {floor:.0f}–{ceiling:.0f}y)."
                )
            rules_fired.append(RuleFiring(
                rule_name="ExperienceRule",
                condition=cond,
                result="FAIL",
                details=fail_reason,
            ))
            is_eligible = False
        else:
            if is_range_mode:
                detail = (
                    f"Within {jd_min_experience:.0f}–{jd_max_experience:.0f}y window "
                    f"(±10% tolerance: {floor:.0f}–{ceiling:.0f}y). Will be scored on D2."
                )
            else:
                detail = f"Within {lower_tolerance:.0f}-year tolerance window. Will be scored on D2."
            rules_fired.append(RuleFiring(
                rule_name="ExperienceRule",
                condition=f"exp={cand_yrs}y",
                result="PASS",
                details=detail,
            ))
    elif is_eligible:
        rules_fired.append(RuleFiring(
            rule_name="ExperienceRule",
            condition="No experience requirement specified",
            result="PASS",
            details="Will be scored on D2.",
        ))

    # ─── Rule 4: Education (specialized stream + 2-level gap) ─
    from ..resume_processing.resume_parser import EDUCATION_ORDINAL
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
            fail_reason = (
                f"Wrong education stream: the role requires a {jd_stream} degree, "
                f"but the candidate's background is {candidate_stream or 'general / non-specialised'}."
            )
        elif required_edu_ord > 0 and candidate_edu_ord > 0:
            gap = required_edu_ord - candidate_edu_ord

            # ─── Strict floor for Masters / PhD requirements ──
            # If the JD asks for Masters (ordinal 4) or above, candidate must
            # have Masters or above. Any gap → NA. This matches the spec
            # "Masters required → only Masters/PhD eligible".
            MASTERS_ORDINAL = 4  # EDUCATION_ORDINAL: PhD=5, Masters=4, Bachelors=3, ...
            if required_edu_ord >= MASTERS_ORDINAL and gap >= 1:
                rules_fired.append(RuleFiring(
                    rule_name="EducationLevelRule",
                    condition=f"JD requires {jd_min_education}+; candidate has only {resume.education_level}",
                    result="FAIL",
                    details=(
                        f"The role requires a {jd_min_education} degree or higher. "
                        f"Candidate's highest qualification is {resume.education_level}, "
                        f"which does not meet the threshold."
                    ),
                ))
                is_eligible = False
                fail_reason = (
                    f"Education below requirement: the role asks for {jd_min_education} "
                    f"or higher, but the candidate has {resume.education_level}."
                )
            elif gap >= 2:
                # Existing rule for non-Masters JDs: 2+ level gap is too far.
                rules_fired.append(RuleFiring(
                    rule_name="EducationLevelRule",
                    condition=f"Gap of {gap} levels below requirement",
                    result="FAIL",
                    details=f"Significant education gap ({resume.education_level} vs {jd_min_education}).",
                ))
                is_eligible = False
                fail_reason = (
                    f"Education too low: the role requires {jd_min_education}, "
                    f"but the candidate has {resume.education_level} "
                    f"(a {gap}-level shortfall)."
                )
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
