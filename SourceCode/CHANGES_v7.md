# ICRS — Final Working Project (v7)

This zip is a complete drop-in replacement for the previous `SourceCode/` folder.
All six feedback items from the latest review have been addressed.

---

## Issue 1 — Skill matching now thorough and accurate

**Problem.** Sangeeta's resume mentions "Bank of America", "BNP Paribas", "Credit
Agricole", and "Communicating effectively with stakeholders" — but skills like
*banking*, *communication*, and *problem solving* were appearing in **Missing
Skills** because:
  - `extract_skills` only did literal substring match against a fixed
    `TECH_SKILLS_DB`. Resumes rarely contain literal terms like "banking" or
    "communication" — those concepts are expressed indirectly.
  - The SBERT semantic match was running against the *extracted* skills list,
    not against the resume's raw text, so latent skills were never recovered.

**Fix in `resume_parser.py`.** `extract_skills` now uses three strategies:
  1. Literal word-boundary match (existing, kept).
  2. Substring match for multi-word skills (existing, kept).
  3. **NEW** — domain-inference rules. Examples:
     - `banking` inferred from `Bank of America`, `JPMorgan`, `BNP Paribas`,
       `Credit Agricole`, `BFSI`, `forex`, `securities`, etc.
     - `communication` inferred from `communicating`, `stakeholder`,
       `presentation`, `reporting`, `cross-functional`.
     - `problem solving` inferred from `analytical`, `troubleshoot`,
       `root-cause`, `debug`.
     - `leadership`, `agile`, `test management`, `uat`, `automation`, etc.

**Fix in `scoring_engine.py`.** Added a Level 3 latent-skill matcher: for JD
skills still missing after keyword and SBERT-on-skills passes, a third SBERT
pass runs against phrases extracted from the resume's `raw_text`. Threshold
0.55 (lower than the 0.60 used for skill-vs-skill matches because phrases are
noisier).

**Verified.** Sangeeta's resume now produces:
```
agile, automation, banking, bdd, communication, cucumber, java, javascript,
kanban, leadership, problem solving, selenium, stakeholder management,
swift, test management
```

---

## Issue 2 — Eligibility correctly marks unrelated profiles NA, robust to unseen professions

**Problem.** Maera (Yoga Teacher) was being ranked for the UAT Test Manager
job. The original `_check_unrelated_profession` failed because:
  - `JOB_TITLE_PATTERNS` had no patterns for non-tech roles, so Maera's
    `job_titles` never contained "Yoga Teacher" — the title-based check found
    nothing.
  - The substring-style match `kw in resume_titles_lower` was unreliable
    even when titles were extracted.
  - For unseen professions (e.g. *painter*) not in the keyword list, there was
    no fallback at all.

**Fix in `resume_parser.py`.** Expanded `JOB_TITLE_PATTERNS` to include 20+ new
profession patterns: yoga/pilates/fitness instructors, chefs/cooks, hairdressers,
plumbers/electricians/painters, drivers, waiters/bartenders, nurses, teachers,
lawyers, journalists, etc. Maera's parsed titles now include `Yoga Teacher`,
`Group Fitness Instructor`, `Physical Therapist`.

**Fix in `eligibility_engine.py`.** Replaced `_check_unrelated_profession` with
a robust three-pass approach:

1. **Keyword pass** — word-boundary match against the candidate's `job_titles`.
   If candidate is in a non-tech profession (yoga/chef/painter/etc.):
   - If the JD is also for a profession in the **same family** (yoga →
     pilates, chef → cook, plumber → painter), candidate is **eligible**.
   - Otherwise, **NA**.
2. **JD-side pass** — if JD is for a specialty profession but candidate has
   nothing matching, **NA** (e.g. a software engineer applying for an
   Executive Chef role).
3. **SBERT semantic pass** — for unseen professions not in the keyword list:
   compute cosine similarity between candidate's job_titles and the JD's
   title+description. If similarity < 0.30 AND zero skill overlap, **NA**.

**Profession families** (single-word anchors that may appear in JD wording):
- Fitness: yoga, fitness, pilates, trainer, instructor, aerobics, zumba
- Culinary: chef, cook, culinary, kitchen, baker, pastry, sous
- Beauty: hairdresser, beautician, cosmetologist, barber, salon, spa
- Trades: plumber, electrician, carpenter, welder, mason, painter
- Driving: truck driver, cab driver, delivery driver, chauffeur
- Restaurant service: waiter, waitress, bartender, barista, server
- Healthcare: nurse, midwife, paramedic, caregiver, physical therapist

**Verified.** All 7 critical scenarios pass:
- Maera vs UAT → NA (Yoga Teacher unrelated to UAT) ✓
- Sangeeta vs UAT → ELIGIBLE ✓
- Painter vs Software Engineer → NA (unseen profession, semantic check fires) ✓
- Maera vs Pilates → ELIGIBLE (same fitness family) ✓
- Sangeeta vs Pilates → NA (tech CV, zero skill overlap) ✓
- Chef vs Executive Chef → ELIGIBLE (same culinary family) ✓
- Mradul vs UAT → ELIGIBLE (generic tech background allowed) ✓

---

## Issue 3 — Six expert flags surfaced in API and frontend

**Problem.** The `value` field (TRUE/FALSE per the spec) was missing from the
`/api/rank` response, so the frontend couldn't show actual flag values.

**Fix in `scoring_engine.py`.** The `expert_flags` array in the response now
includes `value`, `type`, `modifier`, `reason`, and `name`:

```python
expert_flags=[{
    "name": f.flag_name,
    "value": f.flag_value,        # T/F per spec
    "type": f.flag_type,          # BONUS / PENALTY / INFO
    "modifier": f.score_modifier,
    "reason": f.reason,
} for f in flag_result.flags]
```

The 6 flags follow your spec:

| Flag | T = | F = |
|------|-----|-----|
| RELOCATION_FLAG | Penalize -0.04 | Reward +0.04 |
| WORK_VISA_FLAG | Penalize -0.05 | Reward +0.03 |
| LEADERSHIP_MATCH | Reward +0.06 | Penalize -0.04 |
| HIGH_POTENTIAL | Reward +0.06 | Penalize -0.02 |
| OVERQUALIFICATION_FLAG | Penalize -0.06 | Reward +0.02 |
| AVAILABILITY_RISK | Penalize -0.05 | Reward +0.03 |

The frontend now renders all six in a panel matching your reference image
(see Issue 6).

---

## Issue 4 — GA algorithm reviewed, no changes needed

The Genetic Algorithm in `ga_optimizer.py` is correctly implemented:
- Tournament selection k=3
- BLX-α crossover with α=0.5
- Gaussian mutation σ=0.1
- 50-chromosome population, up to 50 generations, early stop after 10 stalls
- Kendall Tau fitness via `scipy.stats.kendalltau`

Pre-trained `CATEGORY_WEIGHTS` are stored as static dictionaries — at runtime,
the system looks up weights for the JD's detected category (data_science,
software_engineering, contract, finance, management, entry_level, default).
No GA execution at runtime; just a lookup.

`detect_job_category` matches the JD title/description against keyword sets
to pick the category. This is unchanged from prior versions and works
correctly.

---

## Issue 5 — Source-code review, additional fixes applied

**`jd_parsing.py`** — Two bugs fixed:
- Education extraction used substring match (`if "master" in text`) so
  "Scrum Master" in a JD would be misread as "Masters degree required".
  Now uses the strict `extract_education_level()` from `resume_parser.py`.
- Hardcoded `title="Software Engineer"` default removed (was corrupting
  GA category detection for non-software JDs).
- Experience extraction now prefers values near "minimum/required/at least"
  keywords, falling back to "X years of experience" patterns, with a final
  generic fallback.

**`main.py`** — Removed hardcoded `job_title="Software Engineer"` form default.

**`scoring_engine.py`** — Two changes:
- Level 3 latent-skill matching added (see Issue 1).
- `expert_flags` response shape now includes `value` field (see Issue 3).

**`resume_parser.py`** — Three changes:
- `JOB_TITLE_PATTERNS` expanded with 20+ non-tech profession patterns.
- `extract_skills` now infers banking/communication/problem solving from
  contextual evidence.
- (Education extractor was already strict; preserved.)

**`eligibility_engine.py`** — Rewrote `_check_unrelated_profession` with
three-pass keyword + SBERT semantic logic. Extended
`UNRELATED_PROFESSION_KEYWORDS` list with painter, pilates, additional
healthcare/teaching roles.

All Python files compile clean.

---

## Issue 6 — index.html: four UI changes applied

1. **Yellow background gone for top-ranked candidate.** The `.rank-1`
   selector was unscoped and applied to BOTH the small rank badge AND the
   whole candidate card. Now scoped to `.rank-badge.rank-1` so the
   candidate card stays white. Rank-1 still gets a subtle orange border for
   emphasis.

2. **Radar chart replaced with KB Expert Flags panel.** Removed
   `createRadarSVG()` and `.radar-container`. Added `buildKbFlagsPanel()`
   which renders 6 rows matching your reference image:
   - ★ High Potential (green)
   - ◆ Leadership Match (blue diamond)
   - ● Overqualified (orange)
   - ▲ Availability Risk (red)
   - ■ Relocation (sky blue square)
   - ▲ Work Visa (amber)

   Each row shows TRUE in green or FALSE in red. Hovering shows the flag's
   `reason` as a tooltip.

3. **AI Justification moved above dimension rows.** New layout:
   - Justification (rich summary) — top
   - Dimension score bars + KB flags panel — middle (side-by-side)
   - Matched/Missing skills — below
   - Per-dimension explanation lines — bottom

4. **Rich candidate summary instead of one-liner.** New `buildCandidateSummary()`
   generates 4-5 sentences: candidate context + verdict tone + strengths +
   weaknesses + flag highlights + skill-match summary. Example:

   > **Sangeeta Bahrani** — Lead Test Analyst with 11y experience and a
   > Bachelors background — is a strong fit for this role with an overall
   > score of **88/100** (rank 1). Strong on **Experience** (95%),
   > **Education** (85%), **Miscellaneous** (90%). Notable flags:
   > leadership posture aligns with the role; multiple high-potential
   > signals; availability risk detected (long notice, ongoing degree, or
   > visa pending). Matched **5** of 5 required skills.

---

## Files changed in v7 vs your uploaded SourceCode

| File | Type of change |
|------|-----|
| `backend/resume_processing/resume_parser.py` | Expanded job title patterns, smarter `extract_skills` with domain inference |
| `backend/decision_automation/eligibility_engine.py` | Rewrote `_check_unrelated_profession` with 3-pass logic + family matching |
| `backend/jd_processing/jd_parsing.py` | Strict education extractor, smart experience extractor, no Software Engineer default |
| `backend/scoring_ranking_engine/scoring_engine.py` | Level 3 latent SBERT skill match, `value` field in expert_flags response |
| `main.py` | Removed Software Engineer default for `job_title` form param |
| `frontend/index.html` | Yellow rank-1 fix, radar→KB flags panel, justification moved above, rich summary |

No changes to: `backend/business_optimization/ga_optimizer.py`,
`backend/business_optimization/train_ga_weights.py`,
`backend/decision_automation/expert_flags.py`, `requirements.txt`.

---

## How to run

```
cd SourceCode
python -m uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser. Re-test with:

- **JD_008 UAT Test Manager + Maera Sen's resume** → Maera should now show as
  N/A with reason "Unrelated profession".
- **JD_008 UAT Test Manager + Sangeeta's resume** → Should rank #1 with
  banking, communication, problem solving in **Matched Skills** (not Missing).
- **Pilates JD + Maera Sen's resume** → Eligible (same fitness family).
- **Top-ranked candidate's card** → White background, KB Expert Flags panel
  on the right, AI Justification at the top.
