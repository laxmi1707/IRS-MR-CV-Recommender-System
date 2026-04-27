import re

from ..resume_processing.resume_parser import extract_skills, EDUCATION_LEVELS, EDUCATION_ORDINAL
from ..scoring_ranking_engine.scoring_engine import JobDescription


def parse_job_description(text: str, title: str = "") -> JobDescription:
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
        description=text,
        required_skills=skills,
        min_experience_years=min_exp,
        required_education=edu_level,
        max_notice_period_days=90,
    )