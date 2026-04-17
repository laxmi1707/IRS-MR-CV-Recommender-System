"""
main.py — FastAPI Backend Server for ICRS / S-Rank
Provides REST API endpoints for resume upload, parsing, and ranking.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import json

from resume_parser import parse_resume
from scoring_engine import (
    rank_candidates,
    parse_job_description,
    DEFAULT_WEIGHTS,
)

# ─── FastAPI App Setup ────────────────────────────────────────
app = FastAPI(
    title="S-Rank ICRS API",
    description="Intelligent Candidate Ranking System — NUS-ISS MTech AI",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────
class RankingResponse(BaseModel):
    success: bool
    message: str
    job_description: dict
    candidates: list[dict]
    total_candidates: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="S-Rank ICRS API",
        version="1.0.0",
    )


@app.post("/api/rank")
async def rank_resumes(
    job_title: str = Form("Software Engineer"),
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
    weights: Optional[str] = Form(None),
):
    """
    Main ranking endpoint.
    
    Accepts:
    - job_title: Title of the position
    - job_description: Full JD text
    - resumes: List of uploaded resume files (PDF/DOCX)
    - weights: Optional JSON string of custom dimension weights
    
    Returns:
    - Ranked list of candidates with scores and explanations
    """
    try:
        # Parse custom weights if provided
        custom_weights = None
        if weights:
            try:
                custom_weights = json.loads(weights)
            except json.JSONDecodeError:
                custom_weights = None

        # Parse job description
        jd = parse_job_description(job_description, title=job_title)

        # Parse all resumes
        parsed_resumes = []
        filenames = []
        for resume_file in resumes:
            file_bytes = await resume_file.read()
            parsed = parse_resume(file_bytes, resume_file.filename)
            parsed_resumes.append(parsed)
            filenames.append(resume_file.filename)

        if not parsed_resumes:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No valid resumes uploaded."},
            )

        # Rank candidates
        rankings = rank_candidates(
            parsed_resumes,
            jd,
            weights=custom_weights,
        )

        # Format response
        candidates = []
        for r in rankings:
            candidates.append({
                "rank": r.rank,
                "name": r.name,
                "email": r.email,
                "overall_score": r.overall_score,
                "dimensions": [
                    {
                        "name": ds.dimension,
                        "score": round(ds.score * 100, 1),
                        "weight": ds.weight,
                        "weighted_score": round(ds.weighted_score * 100, 1),
                        "explanation": ds.explanation,
                    }
                    for ds in r.dimension_scores
                ],
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "experience_years": r.experience_years,
                "education_level": r.education_level,
                "job_titles": r.job_titles,
                "notice_period_days": r.notice_period_days,
                "justification": r.justification,
            })

        return {
            "success": True,
            "message": f"Successfully ranked {len(candidates)} candidates.",
            "job_description": {
                "title": jd.title,
                "required_skills": jd.required_skills[:15],
                "min_experience": jd.min_experience_years,
                "required_education": jd.required_education,
            },
            "candidates": candidates,
            "total_candidates": len(candidates),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error processing resumes: {str(e)}",
            },
        )


@app.post("/api/parse-resume")
async def parse_single_resume(
    resume: UploadFile = File(...),
):
    """Parse a single resume and return structured data (for debugging)."""
    try:
        file_bytes = await resume.read()
        parsed = parse_resume(file_bytes, resume.filename)

        return {
            "success": True,
            "data": {
                "name": parsed.name,
                "email": parsed.email,
                "phone": parsed.phone,
                "skills": parsed.skills,
                "experience_years": parsed.experience_years,
                "education_level": parsed.education_level,
                "job_titles": parsed.job_titles,
                "notice_period_days": parsed.notice_period_days,
                "summary": parsed.summary[:300],
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)},
        )


@app.get("/api/weights")
async def get_default_weights():
    """Return the default dimension weights."""
    return {
        "weights": DEFAULT_WEIGHTS,
        "dimensions": [
            {"key": "technical_skills", "label": "Technical Skills", "description": "SBERT + keyword matching"},
            {"key": "experience", "label": "Experience", "description": "Non-linear log scaling"},
            {"key": "education", "label": "Education", "description": "Ordinal comparison"},
            {"key": "availability", "label": "Availability", "description": "Notice period tiers"},
            {"key": "miscellaneous", "label": "Miscellaneous", "description": "Job title + relevance"},
        ],
    }


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
