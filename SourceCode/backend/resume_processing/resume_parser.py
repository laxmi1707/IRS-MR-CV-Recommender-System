"""
resume_parser.py — Resume Document Parser
Extracts structured information from PDF/DOCX resumes using
regex patterns and spaCy NLP for entity recognition.
"""

import re
import io
from dataclasses import dataclass, field
from typing import Optional

# PDF parsing
import pdfplumber

# DOCX parsing
from docx import Document


@dataclass
class ParsedResume:
    """Structured representation of an extracted resume."""
    raw_text: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    experience_text: str = ""
    education_level: str = ""  # PhD, Masters, Bachelors, Diploma, etc.
    education_text: str = ""
    job_titles: list[str] = field(default_factory=list)
    notice_period_days: int = 90  # default assumption
    certifications: list[str] = field(default_factory=list)
    summary: str = ""


# ─── Section Header Patterns ──────────────────────────────────
SECTION_PATTERNS = {
    "skills": re.compile(
        r"(?i)^(?:technical\s+)?skills|competenc|technologies|proficienc|expertise",
        re.MULTILINE
    ),
    "experience": re.compile(
        r"(?i)^(?:work\s+)?experience|employment|career\s+history|professional\s+background|work\s+history",
        re.MULTILINE
    ),
    "education": re.compile(
        r"(?i)^education|academic|qualification|degree",
        re.MULTILINE
    ),
    "summary": re.compile(
        r"(?i)^(?:professional\s+)?summary|objective|profile|about\s+me|career\s+objective",
        re.MULTILINE
    ),
    "certifications": re.compile(
        r"(?i)^certif|licens|accredit|awards",
        re.MULTILINE
    ),
}

# ─── Skill Extraction Patterns ────────────────────────────────
TECH_SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    # Frameworks
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "node.js", "express", "next.js", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "spark", "hadoop", "airflow", "TestNG", "junit", "BDD", "cucumber","selenium",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "dynamodb", "cassandra", "neo4j", "oracle",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "git", "linux", "ansible", "grafana", "prometheus",
    # AI/ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "transformers", "bert", "gpt", "llm", "reinforcement learning",
    "neural networks", "random forest", "xgboost", "svm",
    "natural language processing", "generative ai", "rag",
    # Data
    "data science", "data engineering", "etl", "data pipeline",
    "power bi", "tableau", "looker", "data visualization",
    "statistics", "a/b testing", "hypothesis testing",
    # Other
    "agile", "waterfall" "scrum", "jira", "rest api", "graphql", "microservices",
    "system design", "oop", "design patterns", "html", "css",
    "banking", "finance", "healthcare", "e-commerce", "cybersecurity", "blockchain","kanban","scrum master", "communication","problem solving",
    
]

# ─── Education Level Mapping ──────────────────────────────────
EDUCATION_LEVELS = {
    "phd": "PhD",
    "ph.d": "PhD",
    "doctorate": "PhD",
    "doctor of": "PhD",
    "masters of": "Masters",
    "mtech": "Masters",
    "m.tech": "Masters",
    "msc": "Masters",
    "m.sc": "Masters",
    "mba": "Masters",
    "m.s.": "Masters",
    "bachelor": "Bachelors",
    "b.tech": "Bachelors",
    "btech": "Bachelors",
    "b.sc": "Bachelors",
    "bsc": "Bachelors",
    "b.e.": "Bachelors",
    "b.eng": "Bachelors",
    "diploma": "Diploma",
    "associate": "Diploma",
    "certificate": "Certificate",
    "high school": "HighSchool",
    "HSC": "HighSchool",
    "secondary": "HighSchool",
    "SSC": "HighSchool",
}

EDUCATION_ORDINAL = {
    "PhD": 5,
    "Masters": 4,
    "Bachelors": 3,
    "Diploma": 2,
    "HighSchool": 1,
    "Unknown": 0,
}

# ─── Job Title Patterns ───────────────────────────────────────
JOB_TITLE_PATTERNS = [
    # ─── Tech / Engineering ───
    r"(?i)(?:senior|junior|lead|principal|staff|chief)?\s*(?:software|data|ml|ai|devops|cloud|full\s*stack|front\s*end|back\s*end|mobile|systems?|security|network)\s*(?:engineer|developer|architect|scientist|analyst|administrator)",
    r"(?i)(?:senior|junior|lead|principal)?\s*(?:product|project|program|engineering)\s*manager",
    r"(?i)(?:senior|junior)?\s*(?:business|data|systems?|research|financial|qa|test)\s*analyst",
    r"(?i)(?:cto|ceo|coo|cfo|vp|director|head)\s+(?:of\s+)?(?:engineering|technology|data|product|operations|sales)",
    r"(?i)(?:technical|team|engineering|tech)\s*lead",
    r"(?i)(?:solutions?|enterprise|technical|cloud|security|data)\s*architect",
    r"(?i)data\s*(?:engineer|scientist|analyst)",
    r"(?i)machine\s*learning\s*engineer",
    r"(?i)research\s*(?:engineer|scientist)",
    r"(?i)\b(?:consultant|intern|trainee|associate|specialist)\b",
    # ─── QA / Testing ───
    r"(?i)(?:senior|junior|lead|principal)?\s*(?:test|qa|quality\s*assurance)\s*(?:lead|manager|engineer|analyst|architect|automation)",
    r"(?i)(?:senior|junior|lead)?\s*(?:automation|sdet|performance)\s*(?:engineer|tester|analyst)",
    r"(?i)test\s*architect",
    r"(?i)uat\s*(?:test\s*)?(?:manager|lead|analyst)",
    # ─── Generic management ───
    r"(?i)(?:senior|junior|lead|head)?\s*(?:program|project|product|account|operations)\s*manager",
    # ─── Non-tech professions (CRITICAL for eligibility detection) ───
    r"(?i)(?:yoga|fitness|pilates|zumba|aerobics|reformer)\s*(?:teacher|instructor|trainer|coach)",
    r"(?i)(?:personal|gym|sports|athletic)\s*trainer",
    r"(?i)(?:executive|head|sous|pastry|line|station)\s*chef",
    r"(?i)\b(?:chef|cook|baker|sommelier)\b",
    r"(?i)\b(?:cashier|waiter|waitress|bartender|barista|hostess?|server)\b",
    r"(?i)(?:hairdresser|hair\s*stylist|barber|beautician|cosmetologist|makeup\s*artist|esthetician|nail\s*technician)",
    r"(?i)(?:plumber|electrician|carpenter|welder|mason|painter|roofer|locksmith|machinist)",
    r"(?i)(?:gardener|florist|landscaper|farmer|farmhand)",
    r"(?i)(?:truck|cab|taxi|delivery|bus|chauffeur)\s*driver",
    r"(?i)(?:security\s*guard|bouncer|doorman|watchman)",
    r"(?i)\b(?:nurse|nursing|caregiver|midwife|paramedic|medic|doctor|surgeon|dentist|veterinarian|pharmacist)\b",
    r"(?i)(?:physical|occupational|speech)\s*therapist",
    r"(?i)group\s*fitness\s*(?:instructor|trainer)",
    r"(?i)(?:retail|store|shop)\s*(?:associate|clerk|assistant)",
    r"(?i)(?:teacher|tutor|professor|lecturer|educator|instructor)",  # education roles
    r"(?i)(?:lawyer|attorney|paralegal|legal\s*assistant|advocate|barrister|solicitor)",
    r"(?i)(?:journalist|reporter|editor|copywriter|content\s*writer)",
]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    # Also extract from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)
    return "\n".join(text_parts)


def extract_sections(text: str) -> dict[str, str]:
    """Split resume text into labelled sections using header patterns."""
    lines = text.split("\n")
    sections = {}
    current_section = "header"
    current_lines = []

    for line in lines:
        matched = False
        for section_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(line.strip()):
                # Save previous section
                sections[current_section] = "\n".join(current_lines)
                current_section = section_name
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    # Save last section
    sections[current_section] = "\n".join(current_lines)
    return sections


def extract_email(text: str) -> str:
    """Extract email address from text."""
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Extract phone number from text."""
    match = re.search(r"[\+]?[(]?\d{1,4}[)]?[-\s./]?\d{3,4}[-\s./]?\d{4}", text)
    return match.group(0) if match else ""


def extract_skills(text: str) -> list[str]:
    """Match known skills against resume text.

    Uses three strategies (in priority order):
      1. Literal word-boundary match of skill in text.
      2. Substring match for multi-word skills.
      3. Domain-inference rules — captures abstract/soft skills that rarely
         appear as literal terms in resumes. Examples:
           - 'banking' inferred from 'Bank of America', 'BNP Paribas', 'JPMorgan'
           - 'communication' inferred from 'communicating effectively with stakeholders'
           - 'problem solving' inferred from 'analytical', 'troubleshooting'

    The third strategy is critical because resumes rarely contain literal
    terms like 'banking' or 'communication' — those are expressed indirectly.
    """
    text_lower = text.lower()
    found_skills = set()

    # Strategy 1 + 2: literal and word-boundary matches
    for skill in TECH_SKILLS_DB:
        skill_l = skill.lower()
        if len(skill_l) <= 3:
            if re.search(r"\b" + re.escape(skill_l) + r"\b", text_lower):
                found_skills.add(skill_l)
        else:
            if skill_l in text_lower:
                found_skills.add(skill_l)

    # Strategy 3: domain-inference — credit a skill if its concept is
    # strongly implied even when the literal word isn't present.
    SKILL_INFERENCE_RULES = {
        "banking": [
            r"(?i)\b(?:bank\s+of\s+\w+|banking|jpmorgan|hsbc|citibank|"
            r"standard\s+chartered|deutsche\s+bank|barclays|bnp\s+paribas|"
            r"credit\s+agricole|abn\s+amro|goldman\s+sachs|morgan\s+stanley|"
            r"wells\s+fargo|state\s+street|swift\s+messag|trade\s+process|"
            r"forex|foreign\s+exchange|securities|bond\s+asset|bfsi|"
            r"capital\s+market|investment\s+bank)\b"
        ],
        "communication": [
            r"(?i)\b(?:communicat|stakeholder|presentation|reporting|liais|"
            r"correspond|interpersonal|verbal\s+and\s+written|cross[-\s]functional)",
        ],
        "problem solving": [
            r"(?i)\b(?:problem[-\s]solv|analytical|troubleshoot|root[-\s]cause|"
            r"debugg|critical[-\s]think|investigat|diagnos)",
        ],
        "leadership": [
            r"(?i)\b(?:leader|managed|mentored|coached|led\s+(?:a|the|team)|"
            r"team\s+lead|head\s+of|directed)",
        ],
        "agile": [
            r"(?i)\b(?:agile|scrum|sprint|kanban|standup|retrospective|"
            r"product\s+owner|backlog\s+groom)",
        ],
        "test management": [
            r"(?i)\b(?:test\s+(?:management|strategy|plan|estimate|architect)|"
            r"qa\s+lead|test\s+lead|defect\s+manage|sign[-\s]off)",
        ],
        "uat": [
            r"(?i)\buat\b|user\s+acceptance\s+test",
        ],
        "automation": [
            r"(?i)\b(?:automat|cucumber|webdriver|rest[-\s]assured|playwright)",
        ],
        "finance": [
            r"(?i)\b(?:finance|financial|treasury|accounting|audit|fund\s+management|"
            r"asset\s+management|portfolio|hedge\s+fund)",
        ],
        "stakeholder management": [
            r"(?i)\b(?:stakeholder|client\s+management|business\s+user|liaison)",
        ],
    }
    for skill, patterns in SKILL_INFERENCE_RULES.items():
        if skill in found_skills:
            continue
        for pat in patterns:
            if re.search(pat, text):
                found_skills.add(skill)
                break

    return list(found_skills)


def extract_experience_years(text: str) -> float:
    """Estimate total years of experience from text patterns.

    Three strategies, in priority order:
      1. Explicit statement like "11 years of experience"
      2. Date ranges with month names: "Aug 2023 - Ongoing", "Jun-2018 - Jan-2020"
      3. Plain year ranges: "2019 - 2023"
    """
    # ─── Pattern 1: explicit "X years of experience" ──────
    matches = re.findall(
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
        text, re.IGNORECASE,
    )
    if matches:
        return min(40.0, max(float(m) for m in matches))

    # ─── Pattern 2: month + year ranges ───────────────────
    # Catches "Aug 2023 - Ongoing", "Jun-2018 - Jan-2020", "Jan 2020 – Jun 2021"
    # Months can be abbreviated; separator can be space or hyphen
    MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    range_re = re.compile(
        rf"{MONTH}[\s\-]*?(\d{{4}})\s*[-–—]\s*"
        rf"(?:{MONTH}[\s\-]*?(\d{{4}})|(Present|Ongoing|Current|Now))",
        re.IGNORECASE,
    )
    total_months = 0
    for m in range_re.finditer(text):
        start_year = int(m.group(1))
        if m.group(2):
            end_year = int(m.group(2))
        else:
            end_year = 2026  # Present/Ongoing
        if 1990 <= start_year <= 2030 and end_year >= start_year:
            total_months += max(0, (end_year - start_year) * 12)

    if total_months > 0:
        return min(40.0, total_months / 12.0)

    # ─── Pattern 3: plain "YYYY - YYYY" ───────────────────
    plain_ranges = re.findall(
        r"(20\d{2})\s*[-–—]\s*(20\d{2}|present|current|ongoing)",
        text, re.IGNORECASE,
    )
    total = 0
    for start, end in plain_ranges:
        start_y = int(start)
        end_y = 2026 if end.lower() in ("present", "current", "ongoing") else int(end)
        total += max(0, end_y - start_y)

    return min(40.0, float(total))


def extract_education_level(text: str) -> str:
    """Determine highest education level mentioned.

    Uses word-boundary regex to avoid:
      - 'scrum master' matching 'master'
      - 'project associate' matching 'associate'
      - 'phpdev' matching 'phd'
      - 'masterclass' matching 'master'

    Also rejects matches whose immediate context is clearly NOT
    an academic degree (e.g. 'scrum master', 'master of ceremonies',
    'master class', 'master degree program' is OK).
    """
    text_lower = text.lower()
    highest = "Unknown"
    highest_ord = -1

    # Words that, when adjacent to "master"/"associate"/etc., indicate
    # a NON-academic context. We reject the match if these are nearby.
    REJECT_NEAR_MASTER = [
        "scrum", "project", "product", "story", "ceremony", "ceremonies",
        "class", "key", "mind", "yoga", "chess", "game",
    ]
    REJECT_NEAR_ASSOCIATE = [
        "project", "junior", "senior", "sales", "marketing", "research",
        "executive", "team",
    ]
    REJECT_NEAR_PHD = [
        # rare false positives, mostly safe
    ]

    # Build patterns with word boundaries
    # Order matters: longer/more-specific keys first
    education_keys = [
        ("phd", "PhD"),
        ("ph.d", "PhD"),
        ("ph\\.d\\.", "PhD"),
        ("doctorate", "PhD"),
        ("doctor of philosophy", "PhD"),
        ("master of", "Masters"),
        ("master's", "Masters"),
        ("masters degree", "Masters"),
        ("master degree", "Masters"),
        ("m\\.tech", "Masters"),
        ("mtech", "Masters"),
        ("m\\.sc", "Masters"),
        ("msc", "Masters"),
        ("m\\.s\\.", "Masters"),
        ("m\\.eng", "Masters"),
        ("meng", "Masters"),
        ("mba", "Masters"),
        ("master", "Masters"),  # fallback — applies REJECT list
        ("bachelor of", "Bachelors"),
        ("bachelor's", "Bachelors"),
        ("bachelors degree", "Bachelors"),
        ("bachelor degree", "Bachelors"),
        ("b\\.tech", "Bachelors"),
        ("btech", "Bachelors"),
        ("b\\.sc", "Bachelors"),
        ("bsc", "Bachelors"),
        ("b\\.e\\.", "Bachelors"),
        ("b\\.eng", "Bachelors"),
        ("beng", "Bachelors"),
        ("bachelor", "Bachelors"),
        ("diploma", "Diploma"),
        ("associate of", "Diploma"),
        ("associate degree", "Diploma"),
        ("associate", "Diploma"),  # applies REJECT list
        ("certificate", "Certificate"),
        ("high school", "HighSchool"),
        ("secondary school", "HighSchool"),
    ]

    for keyword_pat, level in education_keys:
        # word-boundary regex
        pat = r"\b" + keyword_pat + r"\b"
        for m in re.finditer(pat, text_lower):
            # Pull a small window around the match to inspect context
            start = max(0, m.start() - 25)
            end = min(len(text_lower), m.end() + 25)
            context = text_lower[start:end]

            # Apply rejection rules
            matched_word = m.group(0)
            if matched_word == "master":
                if any(reject in context for reject in REJECT_NEAR_MASTER):
                    continue
            if matched_word == "associate":
                if any(reject in context for reject in REJECT_NEAR_ASSOCIATE):
                    continue

            # Accept this match
            ord_val = EDUCATION_ORDINAL.get(level, 0)
            if ord_val > highest_ord:
                highest_ord = ord_val
                highest = level
            break  # one match per keyword is enough

    return highest


def extract_job_titles(text: str) -> list[str]:
    """Extract job titles from resume text."""
    titles = []
    for pattern in JOB_TITLE_PATTERNS:
        matches = re.findall(pattern, text)
        titles.extend(matches)
    # Clean and deduplicate
    cleaned = list(set(t.strip().title() for t in titles if len(t.strip()) > 3))
    return cleaned[:5]  # limit to top 5


def extract_notice_period(text: str) -> int:
    """Extract notice period in days from resume text."""
    text_lower = text.lower()

    # Pattern: "notice period: X days/weeks/months"
    match = re.search(
        r"notice\s*period[:\s]*(\d+)\s*(day|week|month)",
        text_lower
    )
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if "week" in unit:
            return value * 7
        elif "month" in unit:
            return value * 30
        return value

    # Pattern: "immediately available" / "available immediately"
    if re.search(r"immediat(?:ely)?\s*available|available\s*immediat", text_lower):
        return 0

    # Default assumption
    return 90


def parse_resume(file_bytes: bytes, filename: str) -> ParsedResume:
    """
    Main entry point: parse a resume file into structured data.
    Supports PDF and DOCX formats.
    """
    # Step 1: Extract raw text based on file type
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        raw_text = extract_text_from_docx(file_bytes)
    else:
        raw_text = file_bytes.decode("utf-8", errors="ignore")

    if not raw_text.strip():
        return ParsedResume(raw_text="[Empty document]")

    # Step 2: Split into sections
    sections = extract_sections(raw_text)

    # Step 3: Extract structured fields
    header_text = sections.get("header", "")
    skills_text = sections.get("skills", "") + " " + raw_text
    experience_text = sections.get("experience", "")
    education_text = sections.get("education", "")

    # Extract name from first non-empty line of header
    name_candidates = [
        line.strip() for line in header_text.split("\n")
        if line.strip() and not re.search(r"@|http|www|\d{5,}", line)
    ]
    name = name_candidates[0] if name_candidates else "Unknown"
    # Clean name: remove common non-name elements
    name = re.sub(r"(?i)curriculum\s+vitae|resume|cv", "", name).strip()

    # Education: STRONGLY prefer the EDUCATION section.
    # This avoids matching "Scrum Master Certified" in skills as "Masters degree".
    # Only fall back to the full text if no EDUCATION section was detected.
    if education_text and len(education_text.strip()) > 10:
        education_level = extract_education_level(education_text)
        # If the dedicated section yielded nothing, scan summary too (some CVs
        # mention "Master's degree" only in the summary, not under EDUCATION)
        if education_level == "Unknown":
            summary_text = sections.get("summary", "")
            education_level = extract_education_level(education_text + " " + summary_text)
    else:
        # No education section — fall back to full text but with the strict matcher
        education_level = extract_education_level(raw_text)

    resume = ParsedResume(
        raw_text=raw_text,
        name=name[:100],
        email=extract_email(raw_text),
        phone=extract_phone(raw_text),
        skills=extract_skills(skills_text),
        experience_years=extract_experience_years(raw_text),
        experience_text=experience_text[:2000],
        education_level=education_level,
        education_text=education_text[:1000],
        job_titles=extract_job_titles(raw_text),
        notice_period_days=extract_notice_period(raw_text),
        certifications=[],
        summary=sections.get("summary", "")[:500],
    )

    return resume


if __name__ == "__main__":
    # Quick test
    sample = """
    John Doe
    john.doe@email.com | +65 9123 4567

    PROFESSIONAL SUMMARY
    Senior Data Scientist with 8 years of experience in machine learning and NLP.

    TECHNICAL SKILLS
    Python, TensorFlow, PyTorch, SQL, AWS, Docker, Kubernetes, React

    WORK EXPERIENCE
    Senior Data Scientist | Google | 2020 - Present
    - Built NLP pipelines processing 1M documents daily

    Data Scientist | Facebook | 2016 - 2020
    - Developed recommendation engine using deep learning

    EDUCATION
    Master of Science in Computer Science | Stanford University | 2016
    Bachelor of Science in Mathematics | MIT | 2014

    Notice Period: 30 days
    """
    result = parse_resume(sample.encode(), "test.txt")
    print(f"Name: {result.name}")
    print(f"Skills: {result.skills}")
    print(f"Experience: {result.experience_years} years")
    print(f"Education: {result.education_level}")
    print(f"Job Titles: {result.job_titles}")
    print(f"Notice Period: {result.notice_period_days} days")
