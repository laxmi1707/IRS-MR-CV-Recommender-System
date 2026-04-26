# Backend Build And Run Instructions

## Location

Backend root:

`SourceCode/backend`

Main entrypoint:

`SourceCode/backend/main.py`

## Recommended Python Version

Use Python `3.10`, `3.11`, or `3.12`.

Some packages in `requirements.txt`, especially `scikit-learn==1.3.2`, may not install cleanly on Python `3.13` because wheels may be unavailable and pip may try to build from source.

## Create Virtual Environment

From the repository root:

```powershell
cd IRS-MR-CV-Recommender-System
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Backend Dependencies

From the backend folder:

```powershell
cd SourceCode\backend
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Backend

Option 1: Run with uvicorn directly

```powershell
cd SourceCode\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Option 2: Run the Python entrypoint

```powershell
cd SourceCode\backend
python main.py
```

## Verify The Backend Is Running

Open this URL in a browser:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "S-Rank ICRS API",
  "version": "2.0.0",
  "pipeline": "6-step (Eligibility → Flags → Score → GA → Rank → XAI)"
}
```

## Main API Endpoints

- `GET /health`
- `POST /api/rank`
- `POST /api/parse-resume`
- `GET /api/weights`

## Important Folder Layout

Backend modules are imported relative to `SourceCode/backend`, so run commands from inside that folder or use it as the working directory.

- `main.py`
- `resume_processing/`
- `decision_automation/`
- `scoring_ranking_engine/`
- `business_optimization/`

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'fastapi'`:

```powershell
python -m pip install -r requirements.txt
```

If you get package build failures on Python `3.13`, switch to Python `3.10` to `3.12`, recreate the virtual environment, and reinstall dependencies.