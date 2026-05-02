"""
main.py — FastAPI Backend for ICRS 6-Step Pipeline
"""

import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import json

from backend.decision_automation.eligibility_engine import check_eligibility
from backend.jd_processing.jd_parsing import parse_job_description
from backend.resume_processing.resume_parser import parse_resume
from backend.scoring_ranking_engine.scoring_engine import CandidateRanking, get_sbert_model, rank_candidates
from backend.business_optimization.ga_optimizer import CATEGORY_WEIGHTS, detect_job_category

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
    job_title: str = Form(""),
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

        parsed_resumes = []
        for resume_file in resumes:
            file_bytes = await resume_file.read()
            parsed = parse_resume(file_bytes, resume_file.filename or "resume")
            parsed_resumes.append(parsed)

        if not parsed_resumes:
            return JSONResponse(status_code=400,
                content={"success": False, "message": "No valid resumes."})

        sbert_model = get_sbert_model()

        eligible_resumes = []
        eligibility_results = []
        not_applicable = []
        for parsed_resume in parsed_resumes:
            eligibility = check_eligibility(
                resume=parsed_resume,
                jd_required_skills=jd.required_skills,
                jd_min_experience=jd.min_experience_years,
                jd_max_experience=jd.max_experience_years,
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
async def parse_single_resume(resume: UploadFile = File(...)):
    try:
        file_bytes = await resume.read()
        parsed = parse_resume(file_bytes, resume.filename or "resume")
        return {
            "success": True,
            "data": {
                "name": parsed.name, "email": parsed.email,
                "phone": parsed.phone, "skills": parsed.skills,
                "experience_years": parsed.experience_years,
                "education_level": parsed.education_level,
                "job_titles": parsed.job_titles,
                "notice_period_days": parsed.notice_period_days,
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
