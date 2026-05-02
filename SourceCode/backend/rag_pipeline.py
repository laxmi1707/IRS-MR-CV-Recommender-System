import hashlib
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from .resume_processing.resume_parser import ParsedResume, parse_resume
from .vector_store.chroma_resume_store import ChromaResumeStore


VECTOR_DB_DIR = Path(__file__).resolve().parent / "resume_vector_db"
resume_store = ChromaResumeStore(VECTOR_DB_DIR)


def get_vector_db_stats() -> dict:
    return resume_store.get_collection_stats()


def _build_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _build_resume_record(parsed: ParsedResume, file_hash: str, filename: str, candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "name": parsed.name,
        "email": parsed.email,
        "phone": parsed.phone,
        "skills": parsed.skills,
        "experience_years": parsed.experience_years,
        "experience_text": parsed.experience_text,
        "education_level": parsed.education_level,
        "education_text": parsed.education_text,
        "job_titles": parsed.job_titles,
        "notice_period_days": parsed.notice_period_days,
        "certifications": parsed.certifications,
        "career_gaps": parsed.career_gaps,
        "summary": parsed.summary,
        "raw_text": parsed.raw_text,
        "file_hash": file_hash,
        "source_filename": filename,
    }


def _parsed_resume_from_store(details: dict) -> ParsedResume:
    return ParsedResume(
        raw_text=details.get("raw_text", ""),
        name=details.get("name", "Unknown"),
        email=details.get("email", ""),
        phone=details.get("phone", ""),
        skills=details.get("skills", []),
        experience_years=float(details.get("experience_years", 0.0) or 0.0),
        experience_text=details.get("experience_text", ""),
        education_level=details.get("education_level", "Bachelors") or "Bachelors",
        education_text=details.get("education_text", ""),
        job_titles=details.get("job_titles", []),
        notice_period_days=int(details.get("notice_period_days", 90) or 90),
        certifications=details.get("certifications", []),
        career_gaps=details.get("career_gaps", []),
        summary=details.get("summary", ""),
    )


async def store_uploaded_resumes(resumes: Optional[list[UploadFile]]) -> dict:
    stored = []
    duplicates = []

    for resume_file in resumes or []:
        file_bytes = await resume_file.read()
        if not file_bytes:
            continue

        file_hash = _build_file_hash(file_bytes)
        existing_candidate = resume_store.find_candidate_by_file_hash(file_hash)
        if existing_candidate:
            existing_details = resume_store.get_candidate_details(existing_candidate["candidate_id"])
            if existing_details:
                duplicates.append({
                    "candidate_id": existing_details["candidate_id"],
                    "name": existing_details["name"],
                    "source_filename": existing_details.get("source_filename", ""),
                    "status": "already_stored",
                })
            continue

        parsed = parse_resume(file_bytes, resume_file.filename or "resume")
        candidate_id = f"cand_{uuid.uuid4().hex[:12]}"
        resume_store.index_resume(
            _build_resume_record(parsed, file_hash, resume_file.filename or "resume", candidate_id),
            candidate_id,
        )
        stored.append({
            "candidate_id": candidate_id,
            "name": parsed.name,
            "source_filename": resume_file.filename or "resume",
            "status": "stored",
        })

    return {"stored": stored, "duplicates": duplicates}


def get_ingestion_candidate_ids(ingestion: dict) -> list[str]:
    candidate_ids = []
    seen = set()

    for group in (ingestion.get("stored", []), ingestion.get("duplicates", [])):
        for candidate in group:
            candidate_id = candidate.get("candidate_id")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidate_ids.append(candidate_id)

    return candidate_ids


def retrieve_resumes_for_scoring(
    jd,
    top_k: Optional[int] = None,
    candidate_ids: Optional[list[str]] = None,
) -> tuple[list[ParsedResume], list[dict]]:
    collection_stats = resume_store.get_collection_stats()
    candidate_count = collection_stats["total_candidates"]
    if candidate_count == 0:
        return [], []

    if candidate_ids:
        scoped_ids = candidate_ids[:top_k] if top_k else candidate_ids
        parsed_resumes = []
        hydrated_candidates = []

        for candidate_id in scoped_ids:
            details = resume_store.get_candidate_details(candidate_id)
            if not details:
                continue

            parsed_resumes.append(_parsed_resume_from_store(details))
            hydrated_candidates.append({
                "candidate_id": candidate_id,
                "similarity_score": None,
                "source_filename": details.get("source_filename", ""),
            })

        return parsed_resumes, hydrated_candidates

    search_limit = top_k or candidate_count
    query = " ".join(filter(None, [jd.title, jd.description, " ".join(jd.required_skills)]))
    search_results = resume_store.search_resumes(query=query, top_k=search_limit)
    if not search_results:
        return [], []

    parsed_resumes = []
    hydrated_candidates = []
    for result in search_results:
        details = resume_store.get_candidate_details(result["candidate_id"])
        if not details:
            continue
        parsed_resumes.append(_parsed_resume_from_store(details))
        hydrated_candidates.append({
            "candidate_id": result["candidate_id"],
            "similarity_score": round(result["similarity_score"], 4),
            "source_filename": details.get("source_filename", ""),
        })

    return parsed_resumes, hydrated_candidates
