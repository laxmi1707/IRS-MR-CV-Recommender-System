# Backend Build And Run Instructions

## Location

Backend root:

`SourceCode/backend`

Main entrypoint:

`SourceCode/backend/main.py`

## Recommended Python Version

This project uses Python `3.12`.

Use Python `3.12` for local development, dependency installation, and running the backend.

Some packages in `requirements.txt`, especially `scikit-learn==1.3.2`, may not install cleanly on Python `3.13` because wheels may be unavailable and pip may try to build from source.

## Create Virtual Environment

From the repository root:

```powershell
cd IRS-MR-CV-Recommender-System
py -3.12 -m venv .venv
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

If dependency installation fails on Windows with an error similar to:

```text
OSError: [Errno 2] No such file or directory
```

and the path includes a deep package path such as `transformers`, use a short drive mapping and retry:

```powershell
subst X: "C:\full\path\to\IRS-MR-CV-Recommender-System"
cd X:\SourceCode\backend
X:\.venv\Scripts\python.exe -m pip install --upgrade pip
X:\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This avoids Windows path-length issues during package extraction.

## Run The Backend

Option 1: Run with uvicorn directly

```powershell
cd SourceCode\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Use this option when you want auto-reload during development.

Option 2: Run the Python entrypoint

```powershell
cd SourceCode\backend
python main.py
```

Use this option for a direct non-reload launch of the backend.

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

If you get a long-path install error on Windows while installing `transformers` or another deep dependency tree, use a short drive mapping with `subst` and rerun the install from that mapped drive.