"""main.py — FastAPI Backend for ICRS 6-Step Pipeline."""

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from decision_automation.eligibility_engine import check_eligibility
from jd_processing.jd_parsing import parse_job_description
from rag_database.chroma_store import ChromaResumeStore
from resume_processing.resume_parser import (
    ParsedResume,
    ResumeParseError,
    SUPPORTED_RESUME_EXTENSIONS,
    parse_resume,
    parsed_resume_from_dict,
    parsed_resume_to_dict,
)
from scoring_ranking_engine.scoring_engine import CandidateRanking, get_sbert_model, rank_candidates
from business_optimization.ga_optimizer import CATEGORY_WEIGHTS, detect_job_category

app = FastAPI(
    title="S-Rank ICRS API",
    description="Intelligent Candidate Ranking System — 6-Step Pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

resume_store: Optional[ChromaResumeStore] = None


def get_resume_store() -> ChromaResumeStore:
    global resume_store
    if resume_store is None:
        resume_store = ChromaResumeStore()
    return resume_store


def _build_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _build_ingested_resume_payload(record: dict, status: str) -> dict:
    return {
        **record,
        "status": status,
    }


async def ingest_uploaded_resumes(resume_files: list[UploadFile]) -> tuple[list[ParsedResume], list[dict], list[dict]]:
    parsed_resumes: list[ParsedResume] = []
    stored_resumes: list[dict] = []
    failed_files: list[dict] = []
    store = get_resume_store()

    for resume_file in resume_files:
        filename = resume_file.filename or "resume.pdf"
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_RESUME_EXTENSIONS:
            failed_files.append({
                "filename": filename,
                "error": "Unsupported file type. Upload PDF, DOCX, or DOC files.",
            })
            continue

        file_bytes = await resume_file.read()
        if not file_bytes:
            failed_files.append({
                "filename": filename,
                "error": "Uploaded file is empty.",
            })
            continue

        file_hash = _build_file_hash(file_bytes)
        existing_record = store.find_resume_by_file_hash(file_hash)
        if existing_record:
            parsed_resumes.append(parsed_resume_from_dict(existing_record))
            stored_resumes.append(_build_ingested_resume_payload(existing_record, "already_stored"))
            continue

        try:
            parsed_resume = parse_resume(file_bytes, filename)
            stored_record = store.upsert_parsed_resume(parsed_resume, filename, file_hash)
        except ResumeParseError as exc:
            failed_files.append({"filename": filename, "error": str(exc)})
            continue
        except Exception as exc:
            failed_files.append({"filename": filename, "error": f"Failed to store parsed resume: {exc}"})
            continue

        parsed_resumes.append(parsed_resume)
        stored_resumes.append(_build_ingested_resume_payload(stored_record, "stored"))

    return parsed_resumes, stored_resumes, failed_files


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "S-Rank ICRS API",
        "version": "2.0.0",
        "pipeline": "6-step (Eligibility → Flags → Score → GA → Rank → XAI)",
    }


@app.post("/api/rank")
async def rank_resumes(
    job_title: str = Form("Software Engineer"),
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
    weights: Optional[str] = Form(None),
):
    try:
        custom_weights = None
        if weights:
            try:
                custom_weights = json.loads(weights)
            except json.JSONDecodeError:
                custom_weights = None

        jd = parse_job_description(job_description, title=job_title)

        parsed_resumes, stored_resumes, failed_files = await ingest_uploaded_resumes(resumes)

        if not parsed_resumes:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No valid resumes were parsed from the uploaded files.",
                    "failed_files": failed_files,
                },
            )

        sbert_model = get_sbert_model()

        eligible_resumes = []
        eligibility_results = []
        not_applicable = []
        for parsed_resume in parsed_resumes:
            eligibility = check_eligibility(
                resume=parsed_resume,
                jd_required_skills=jd.required_skills,
                jd_min_experience=jd.min_experience_years,
                jd_min_education=jd.required_education,
                jd_text=jd.description,
                jd_title=jd.title,
                sbert_model=sbert_model,
            )

            if eligibility.is_eligible:
                eligible_resumes.append(parsed_resume)
                eligibility_results.append(eligibility)
                continue

            not_applicable.append(CandidateRanking(
                name=parsed_resume.name,
                email=parsed_resume.email,
                overall_score=0.0,
                rank=-1,
                is_eligible=False,
                eligibility_reason=eligibility.reason,
                eligibility_trace=eligibility.reasoning_trace,
                experience_years=parsed_resume.experience_years,
                education_level=parsed_resume.education_level,
                job_titles=parsed_resume.job_titles,
                notice_period_days=parsed_resume.notice_period_days,
                justification=f"NOT APPLICABLE: {eligibility.reason}",
                reasoning_chain=eligibility.reasoning_trace,
            ))

        rankings = rank_candidates(
            eligible_resumes,
            jd,
            custom_weights,
            eligibility_results=eligibility_results,
        ) + not_applicable

        # Separate eligible and NA candidates
        eligible = [r for r in rankings if r.is_eligible]
        not_applicable = [r for r in rankings if not r.is_eligible]

        candidates = []
        for r in rankings:
            candidates.append({
                "rank": r.rank,
                "name": r.name,
                "email": r.email,
                "overall_score": r.overall_score,
                "is_eligible": r.is_eligible,
                "eligibility_reason": r.eligibility_reason,
                "dimensions": [
                    {
                        "name": ds.dimension,
                        "score": round(ds.score * 100, 1),
                        "weight": ds.weight,
                        "weighted_score": round(ds.weighted_score * 100, 1),
                        "explanation": ds.explanation,
                    }
                    for ds in r.dimension_scores
                ] if r.is_eligible else [],
                "expert_flags": r.expert_flags,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "experience_years": r.experience_years,
                "education_level": r.education_level,
                "job_titles": r.job_titles,
                "notice_period_days": r.notice_period_days,
                "justification": r.justification,
                "ga_category": r.ga_category,
                "ga_weights": r.ga_weights,
                # XAI reasoning chain
                "eligibility_trace": r.eligibility_trace,
                "flags_trace": r.flags_trace,
                "reasoning_chain": r.reasoning_chain,
            })

        ga_category = detect_job_category(jd.title, jd.description)

        return {
            "success": True,
            "message": f"Processed {len(parsed_resumes)} resumes. "
                       f"{len(eligible)} eligible, {len(not_applicable)} filtered out.",
            "parsed_resumes": [
                {
                    **parsed_resume_to_dict(parsed_resume),
                    "candidate_id": stored_resumes[index].get("candidate_id", ""),
                    "source_filename": stored_resumes[index].get("source_filename", ""),
                    "status": stored_resumes[index].get("status", "stored"),
                }
                for index, parsed_resume in enumerate(parsed_resumes)
            ],
            "failed_files": failed_files,
            "pipeline": {
                "step1_eligibility": f"{len(not_applicable)} candidates filtered as NA",
                "step2_flags": "Expert flags assigned to eligible candidates",
                "step3_scoring": "5-dimensional scoring applied",
                "step4_ga_weights": {
                    "category": ga_category,
                    "weights": CATEGORY_WEIGHTS.get(ga_category, CATEGORY_WEIGHTS["default"]),
                },
                "step5_ranking": f"Best-First Search ranked {len(eligible)} candidates",
                "step6_xai": "Reasoning traces generated for all candidates",
            },
            "job_description": {
                "title": jd.title,
                "required_skills": jd.required_skills[:15],
                "min_experience": jd.min_experience_years,
                "required_education": jd.required_education,
            },
            "candidates": candidates,
            "total_candidates": len(parsed_resumes),
            "eligible_count": len(eligible),
            "na_count": len(not_applicable),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"})


@app.post("/api/parse-resume")
async def parse_uploaded_resumes(resumes: list[UploadFile] = File(...)):
    try:
        parsed_resumes, stored_resumes, failed_files = await ingest_uploaded_resumes(resumes)
        if not parsed_resumes:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No valid resumes were parsed from the uploaded files.",
                    "failed_files": failed_files,
                },
            )

        stored_count = sum(1 for resume in stored_resumes if resume.get("status") == "stored")
        already_stored_count = sum(1 for resume in stored_resumes if resume.get("status") == "already_stored")

        return {
            "success": True,
            "message": f"Parsed {len(parsed_resumes)} resumes and stored them in the RAG database.",
            "resumes": [
                {
                    **parsed_resume_to_dict(parsed_resume),
                    "candidate_id": stored_resumes[index].get("candidate_id", ""),
                    "source_filename": stored_resumes[index].get("source_filename", ""),
                    "file_hash": stored_resumes[index].get("file_hash", ""),
                    "status": stored_resumes[index].get("status", "stored"),
                }
                for index, parsed_resume in enumerate(parsed_resumes)
            ],
            "failed_files": failed_files,
            "counts": {
                "parsed": len(parsed_resumes),
                "stored": stored_count,
                "already_stored": already_stored_count,
                "failed": len(failed_files),
            },
        }
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"success": False, "message": str(e)})


@app.get("/api/weights")
async def get_all_weights():
    return {"categories": CATEGORY_WEIGHTS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
