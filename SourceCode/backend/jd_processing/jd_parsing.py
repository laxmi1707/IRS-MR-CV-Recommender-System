import re

from ..resume_processing.resume_parser import extract_skills, extract_education_level, EDUCATION_ORDINAL
from ..scoring_ranking_engine.scoring_engine import JobDescription


def parse_job_description(text: str, title: str = "") -> JobDescription:
    skills = extract_skills(text)

    # ─── Experience: detect RANGES first (e.g. "4-11 years", "7 to 10 years") ──
    # Range pattern: two numbers separated by - or 'to', then 'years'/'yrs'
    range_match = re.search(
        r"(\d{1,2})\s*(?:[-–—]|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
        text, re.IGNORECASE,
    )
    min_exp = 0.0
    max_exp = 0.0
    if range_match:
        a, b = int(range_match.group(1)), int(range_match.group(2))
        min_exp = float(min(a, b))
        max_exp = float(max(a, b))
    else:
        # Single value: prefer values near 'minimum/required/at least' keywords
        qualified = re.search(
            r"(?:minimum|min\.?|at\s+least|require[ds]?|requires?)[^.]{0,30}?(\d{1,2})\+?\s*(?:years?|yrs?)",
            text, re.IGNORECASE,
        )
        if qualified:
            min_exp = float(qualified.group(1))
        else:
            exp_match = re.search(
                r"(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
                text, re.IGNORECASE,
            )
            if exp_match:
                min_exp = float(exp_match.group(1))
            else:
                generic = re.search(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
                min_exp = float(generic.group(1)) if generic else 3.0

    # Use the strict word-boundary education extractor (avoids 'scrum master' false positives)
    detected = extract_education_level(text)
    edu_level = detected if detected != "Unknown" else "Bachelors"

    return JobDescription(
        title=title or "",
        description=text,
        required_skills=skills,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        required_education=edu_level,
        max_notice_period_days=90,
    )