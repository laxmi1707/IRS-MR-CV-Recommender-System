# Improvement Notes

This document summarizes the improvements implemented in the current branch, the code areas changed, and the reason each change was made.

## Scope

The improvements fall into four areas:

1. Resume parsing robustness
2. Skill extraction quality
3. Job title extraction quality
4. Resume ingestion and RAG persistence

## Files Changed

- `.gitignore`
- `SourceCode/backend/main.py`
- `SourceCode/backend/requirements.txt`
- `SourceCode/backend/resume_processing/resume_parser.py`
- `SourceCode/backend/rag_database/__init__.py`
- `SourceCode/backend/rag_database/chroma_store.py`

## 1. Resume Parsing Robustness

### Code changes

File: `SourceCode/backend/resume_processing/resume_parser.py`

- Added `ResumeParseError` as a dedicated parser exception.
- Added `SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx", ".doc"}`.
- Wrapped `.docx` parsing in `try/except zipfile.BadZipFile` to detect invalid or mislabeled Word files.
- Added `extract_text_from_doc()` for best-effort legacy `.doc` parsing.
- Updated `parse_resume()` to:
  - validate file extension before parsing
  - route parsing by extension (`.pdf`, `.docx`, `.doc`)
  - raise explicit errors for unsupported files, unreadable files, and empty extracted text

### Why this was changed

The parser previously assumed `.docx` files were valid Word packages and treated `.doc` files the same way. That breaks on mislabeled files and older documents. The new logic makes parsing failures explicit and prevents silent bad parses.

### Outcome

- Invalid `.docx` uploads now fail with a clear error message.
- Legacy `.doc` files have a fallback path instead of immediate failure.
- API consumers can distinguish unsupported input from parser failure.

## 2. Skill Extraction Improvements

### Code changes

File: `SourceCode/backend/resume_processing/resume_parser.py`

- Added `_extract_skill_candidates(skills_section)`.
- Updated `extract_skills()` to combine:
  - exact matching from `TECH_SKILLS_DB`
  - phrase extraction from the explicit skills section
- Changed `parse_resume()` to pass both:
  - `skills_text`
  - `skills_section`
- Added handling for:
  - labeled lines such as `Hard Skills:` and `Props Techniques:`
  - comma-, bullet-, semicolon-, and pipe-separated skills
  - wrapped short continuation lines in PDFs
  - duplicate removal with stable output ordering via `sorted(found_skills)`

### Why this was changed

The earlier implementation only returned skills that already existed in the hardcoded `TECH_SKILLS_DB`. That caused resumes with valid non-database skills to return empty or incomplete skill lists, especially in PDFs and domain-specific resumes.

### Outcome

- Skill extraction now works for explicit skill sections even when the skills are not in the hardcoded tech dictionary.
- Wrapped PDF skills like `First Aid` are preserved correctly instead of being split into incorrect fragments.
- The parser now returns richer skill lists for non-standard resumes.

### Example impact

For `Maera Sen Resume.pdf`, the parser now extracts skills such as:

- `yoga`
- `vinyasa`
- `pranayama`
- `hands-on adjustment`
- `first aid`

These were not captured before.

## 3. Job Title Extraction Improvements

### Code changes

File: `SourceCode/backend/resume_processing/resume_parser.py`

- Expanded `JOB_TITLE_PATTERNS` to cover additional title families:
  - executive roles such as `Chief Technology Officer`
  - QA/test roles such as `Lead Test Analyst`, `Test Architect`, `QA Lead`
  - support roles such as `Technical Support Engineer`
  - fitness and healthcare roles such as `Yoga Teacher`, `Group Fitness Instructor`, `Physical Therapist`, `Clinical Exercise Physiologist`
- Added `TITLE_NORMALIZATIONS` for fused or whitespace-damaged resume text, including:
  - `leadtestanalyst`
  - `chieftechnologyofficer`
  - `qalead`
  - `seniortestanalyst`
  - `testarchitect`
  - `teamlead`
- Reworked `extract_job_titles()` to:
  - normalize text before matching
  - use `re.finditer()` instead of `re.findall()`
  - replace unordered `set()` deduplication with ordered deduplication
  - prioritize titles found in the header window
  - prioritize titles near recent-role markers such as `Present`, `Current`, and `2023-2026`

### Why this was changed

The earlier title extractor had three weaknesses:

1. It recognized only a narrow set of mostly software/data titles.
2. It scanned the whole resume and mixed old roles with the primary current role.
3. It used `set()` deduplication, which destroyed title order and made the first returned title unreliable.

### Outcome

- Primary titles are now ranked more sensibly.
- Header titles are favored over incidental titles from later experience entries.
- Fused PDF titles can now be recovered.
- Recent roles are ranked ahead of older roles when both are present.

### Example impact

- `cd_007_KL_CTO.docx` now extracts `Chief Technology Officer` first.
- `Maera Sen Resume.pdf` now extracts `Yoga Teacher` first.
- `Rezume_SangeetaBahrani.pdf` now extracts cleaned titles such as `Lead Test Analyst`, `Qa Lead`, `Senior Test Analyst`, `Test Architect`, and `Team Lead`.

## 4. Parsed Resume Serialization Helpers

### Code changes

File: `SourceCode/backend/resume_processing/resume_parser.py`

- Added `parsed_resume_to_dict()`.
- Added `parsed_resume_from_dict()`.

### Why this was changed

Parsed resumes now need to be stored and retrieved from the RAG layer. These helpers provide a stable conversion between in-memory `ParsedResume` objects and persisted dictionary records.

### Outcome

- Parsed resumes can be stored in Chroma and reconstructed reliably.
- API responses can return parsed resume structures directly.

## 5. RAG Persistence for Parsed Resumes

### Code changes

Files:

- `SourceCode/backend/rag_database/__init__.py`
- `SourceCode/backend/rag_database/chroma_store.py`

Added `ChromaResumeStore` to persist parsed resumes into a local Chroma collection.

Key features:

- resume lookup by file hash
- deterministic candidate ID generation
- embedding text generation from structured resume fields
- metadata storage for skills, job titles, certifications, and candidate details
- reconstruction of stored records back into `ParsedResume`

### Why this was changed

The previous flow parsed resumes per request but did not persist parsed structures for reuse. That caused repeated parsing and prevented using parsed resumes as a local retrieval layer.

### Outcome

- Parsed resumes can now be reused instead of reparsed every time.
- Duplicate uploads can be recognized by file hash.
- Structured parsed resumes are now available for downstream retrieval workflows.

## 6. API Ingestion Improvements

### Code changes

File: `SourceCode/backend/main.py`

- Added `get_resume_store()` singleton accessor.
- Added `_build_file_hash()`.
- Added `_build_ingested_resume_payload()`.
- Added `ingest_uploaded_resumes()` to centralize upload parsing, deduplication, and storage.
- Updated `/rank_resumes` flow to:
  - parse all uploaded resumes through the shared ingestion path
  - return `failed_files`
  - include parsed resume structures and storage status in the response
- Updated `/api/parse-resume` to support multiple uploaded resumes instead of a single file.

### Why this was changed

Resume parsing and storage logic was duplicated and incomplete. The new ingestion path centralizes validation, deduplication, persistence, and error reporting.

### Outcome

- Multiple uploaded resumes can be processed in one request.
- Duplicate resumes are recognized as `already_stored`.
- API responses now expose both successful parses and failed files.

## 7. Dependency and Repository Updates

### Code changes

Files:

- `SourceCode/backend/requirements.txt`
- `.gitignore`

Changes made:

- Added `chromadb==0.5.5` to backend dependencies.
- Ignored temporary Office lock files using `~$*`.
- Ignored local Chroma persistence directory `SourceCode/backend/rag_database/chroma_db/`.

### Why this was changed

The new parsed-resume persistence flow depends on Chroma. The ignore file was also updated so temporary Office files and local vector-store data do not pollute version control.

## Validation Performed

The following checks were run during implementation:

- direct parser execution on multiple sample resumes in `dataset/resumes`
- validation of extracted skills for `Maera Sen Resume.pdf`
- validation of extracted job titles for representative resumes such as:
  - `cd_007_KL_CTO.docx`
  - `cd_009_LBC_CV.docx`
  - `cd_011_TB_SolutionArchitect.docx`
  - `Rezume_SangeetaBahrani.pdf`
- file-level error checks on `resume_parser.py`

## Summary

The main functional improvements are:

- safer and more explicit resume parsing
- richer skill extraction beyond the fixed tech database
- stronger and better-ranked job title extraction
- persistent parsed resume storage in Chroma
- improved API handling for multi-file ingestion and duplicate detection

These changes improve both parsing accuracy and backend workflow reliability.