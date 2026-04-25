"""
COMPREHENSIVE IMPLEMENTATION GUIDE
Intelligent Resume Ranking System (ICRS) — 6-Step Pipeline

============================================================================
PART 1: GA WEIGHT OPTIMIZATION RESULTS
============================================================================

Ground Truth Data Source:
- File: GroundTruth_Data_Candidate Ranking with JD Id.xlsx
- Total Candidates: 120
- Eligible: 63 (52.5%)
- Not Applicable: 57 (47.5%)
- JD Positions: 8 job descriptions

Training Process:
- Extracted 5-dimensional scores (skills, experience, education, availability, misc)
- Normalized scores to 0-1 range from ground truth
- Ran GA optimization per JD with Kendall Tau rank correlation as fitness metric
- Aggregated results by job category using mean averaging

OPTIMIZED CATEGORY WEIGHTS (Calibrated to Ground Truth):
───────────────────────────────────────────────────────────────────────────

1. DATA_SCIENCE (Software engineers, ML, data roles)
   ├─ Technical Skills: 0.2252 (22.5%)
   ├─ Experience: 0.1833 (18.3%)
   ├─ Education: 0.1842 (18.4%)
   ├─ Availability: 0.2283 (22.8%)
   └─ Miscellaneous: 0.1790 (17.9%)
   
   Interpretation: Strong emphasis on technical skills and availability for these
   fast-paced roles. Education and experience equally important.

2. MANAGEMENT (CTO, Director, Engineering Lead roles)
   ├─ Technical Skills: 0.1350 (13.5%)
   ├─ Experience: 0.2091 (20.9%)
   ├─ Education: 0.2102 (21.0%)
   ├─ Availability: 0.1369 (13.7%)
   └─ Miscellaneous: 0.3088 (30.9%)  ← HIGH
   
   Interpretation: Leadership qualities and soft skills (miscellaneous) are
   PARAMOUNT. Experience and education nearly equally weighted. Quick availability
   less critical for senior roles.

3. FINANCE (Accounting, banking, audit, treasury roles)
   ├─ Technical Skills: 0.1645 (16.5%)
   ├─ Experience: 0.3234 (32.3%)  ← HIGHEST
   ├─ Education: 0.1379 (13.8%)
   ├─ Availability: 0.2455 (24.6%)
   └─ Miscellaneous: 0.1288 (12.9%)
   
   Interpretation: EXPERIENCE is critical in finance. Availability ranked 2nd.
   Technical skills and education less emphasized (trust and experience matter
   more than certifications).

4. ENTRY_LEVEL (Junior roles, fresh graduates, internship-oriented)
   ├─ Technical Skills: 0.1557 (15.6%)
   ├─ Experience: 0.1564 (15.6%)
   ├─ Education: 0.2190 (21.9%)
   ├─ Availability: 0.2304 (23.0%)
   └─ Miscellaneous: 0.2386 (23.9%)
   
   Interpretation: Education, availability, and soft skills dominate. Experience
   and technical skills equally de-emphasized (not expected from juniors).

5. DEFAULT (Generic/unclassified roles)
   ├─ Technical Skills: 0.1852 (18.5%)
   ├─ Experience: 0.2538 (25.4%)
   ├─ Education: 0.1152 (11.5%)
   ├─ Availability: 0.2618 (26.2%)
   └─ Miscellaneous: 0.1839 (18.4%)
   
   Interpretation: Experience and availability are primary drivers. Education
   de-emphasized. Balanced across other dimensions.

============================================================================
PART 2: EXPERT FLAGS — PENALTIES & BONUSES
============================================================================

All flags are applied to specific dimensions (skills, experience, education,
availability, miscellaneous) and modify scores by the amount shown.

PENALTY FLAGS (Negative Impact):
───────────────────────────────────────────────────────────────────────────

1. OVERQUALIFICATION_FLAG
   Dimension: experience
   Penalty: -0.06 (-6%)
   Trigger: Candidate has 2.5x+ experience than JD requires
   Example: Senior applying to entry-level role
   Reasoning: Flight risk — will leave for better opportunity
   Confidence: 70%

2. OVERQUALIFICATION_EDU
   Dimension: education
   Penalty: -0.04 (-4%)
   Trigger: Candidate has 5+ levels education vs JD requires ≤3 levels
   Example: PhD applying for Bachelor's-level role
   Reasoning: May be overqualified, potential retention risk
   Confidence: 60%

3. LEADERSHIP_MISMATCH
   Dimension: miscellaneous
   Penalty: -0.02 (-2%)
   Trigger: Candidate has leadership background but JD is non-senior
   Example: Manager applying for IC (Individual Contributor) role
   Reasoning: May not be satisfied in non-leadership position
   Confidence: 50%

4. CAREER_GAP_FLAG
   Dimension: miscellaneous
   Penalty: -0.04 (-4%)
   Trigger: Resume shows career break/gap >12 months OR years in resume show 3+ year gaps
   Example: Candidate was out of workforce for 18 months
   Reasoning: Skills may have atrophied; re-integration risk
   Confidence: 60%

5. CAREER_STABILITY_FLAG
   Dimension: miscellaneous
   Penalty: -0.05 (-5%)
   Trigger: 4+ job titles in ≤5 years of experience
   Example: Job-hopper (changed jobs every 15 months)
   Reasoning: Pattern of instability; may leave quickly
   Confidence: 65%

6. EDUCATION_GAP_FLAG
   Dimension: education
   Penalty: -0.03 × gap_levels (-3% per level)
   Trigger: Candidate education is 1+ levels below JD requirement
   Example: Diploma when Bachelor's required → -3%
   Example: Diploma when Master's required → -6%
   Reasoning: May struggle with academic material expectations
   Confidence: 80%

7. WORK_VISA_FLAG
   Dimension: availability
   Penalty: -0.03 (-3%)
   Trigger: Candidate needs visa sponsorship BUT JD doesn't offer it
   Example: Indian candidate needs H1B for US role with no sponsorship mention
   Reasoning: May not be able to join; legal/processing delays
   Confidence: 50%

8. AVAILABILITY_RISK_FLAG
   Dimension: availability
   Penalty: -0.04 (-4%)
   Trigger: 2+ of: (1) pursuing degree, (2) notice >90 days, (3) visa needed, (4) unstable history
   Example: Currently doing MTech + long notice period + visa needed
   Reasoning: Multiple factors making actual start date uncertain
   Confidence: 65%

───────────────────────────────────────────────────────────────────────────

BONUS FLAGS (Positive Impact):
───────────────────────────────────────────────────────────────────────────

1. LEADERSHIP_MATCH
   Dimension: miscellaneous
   Bonus: +0.08 (+8%)
   Trigger: Candidate has leadership experience AND JD is senior/leadership role
   Example: Team lead applying for CTO role
   Reasoning: Directly relevant experience
   Confidence: 85%

2. EXACT_TITLE_MATCH
   Dimension: miscellaneous
   Bonus: +0.07 (+7%)
   Trigger: Candidate held exact job title as JD
   Example: Previously "Senior Backend Engineer", applying for "Senior Backend Engineer"
   Reasoning: Proven exact-role fit
   Confidence: 90%

3. CERTIFICATION_SCORE
   Dimension: technical_skills
   Bonus: +0.03 per cert (capped at 3 certs → +0.09 max) (+9% max)
   Examples of recognized certs:
   - Cloud: AWS certified, Azure certified, Google Cloud certified, CKA, CKAD
   - Data: TensorFlow certified, Databricks certified, Snowflake certified
   - Professional: PMP, CISSP, ITIL, TOGAF, Prince2, Six Sigma
   - Finance: CFA, CPA, ACCA
   - Testing: ISTQB
   Reasoning: Demonstrates commitment to professional development
   Confidence: 85%

4. UPSKILL_FLAG
   Dimension: technical_skills
   Bonus: +0.03 (+3%)
   Trigger: 2+ evidence of continuous learning (Coursera, Udemy, EDx, MOOCs, self-taught)
   Example: Coursera ML course + Udemy Python course
   Reasoning: Shows initiative and adaptability
   Confidence: 70%

5. RELOCATION_FLAG
   Dimension: availability
   Bonus: +0.03 (+3%)
   Trigger: Resume explicitly states "willing to relocate"
   Example: "Open to relocation"
   Reasoning: Removes geographic barrier
   Confidence: 80%

6. WORK_VISA_FLAG (Positive)
   Dimension: availability
   Bonus: +0.02 (+2%)
   Trigger: JD explicitly offers visa sponsorship AND candidate may need it
   Example: JD says "We sponsor visas" for international role
   Reasoning: Removes visa uncertainty
   Confidence: 60%

7. HIGH_POTENTIAL_FLAG
   Dimension: miscellaneous
   Bonus: +0.05 (+5%)
   Trigger: 2+ indicators: (1) 8+ years exp + 4+ education level, (2) 2+ certifications,
            (3) leadership + advanced education, (4) keywords ("rising star", "top talent")
   Example: Young professional with CTO title, AWS + Azure certs, MBA pursuing
   Reasoning: Demonstrates growth trajectory; likely to add value beyond role
   Confidence: 75%

8. FRESH_GRADUATE (Info only, no modifier)
   Dimension: experience
   Bonus: 0.0
   Trigger: Experience ≤1 year
   Reason: Marker for context — typically used with entry_level weights
   Confidence: 75%

============================================================================
PART 3: 5-DIMENSIONAL SCORING DEFINITIONS
============================================================================

D1: TECHNICAL SKILLS (0.0 - 1.0)
─────────────────────────────────
Measures: Overlap between candidate skills and JD required skills

Methodology:
1. Keyword Matching: Direct skill overlap (Python, Java, React, etc.)
2. Semantic Matching: SBERT embeddings for near-synonyms (threshold 0.65)
   - Example: "Machine Learning" ≈ "ML"
3. Apriori Rules: Implied skills from co-occurrence patterns
   - Example: If resume has Python + pandas, boost for numpy even if not listed
4. Relevance Boost: Overall document similarity (0.25 weight)
   - Prevents harsh penalties for tool/framework mismatch

Score Range: 0.40 - 1.0 (floor at 0.40 to avoid extreme penalties)

Floor at 40% ensures:
- Different tech stack but relevant skills still get credit
- Pure domain mismatches still score >0 (ranked lower, not eliminated)

Examples:
- 12/12 skills matched: 1.0
- 8/12 skills + 2 semantic: 0.83
- 3/12 skills + relevance: 0.65
- 0/12 skills but relevant domain: 0.40

───────────────────────────────────────────────────────────────────────────

D2: EXPERIENCE (0.0 - 1.0)
────────────────────────
Measures: Years of experience vs requirement

Methodology: Bell curve around requirement
- Below requirement: Lenient (min 50%)
- At requirement: 1.0 (optimal)
- 1-1.5x requirement: 0.95-1.0 (acceptable)
- 1.5-2.5x requirement: 0.85-0.95 (good but slightly worried)
- 2.5x+ requirement: 0.55-0.85 (overqualified, flight risk)

Formula:
- If actual ≤ required: score = 0.50 + (ratio^0.7 * 0.50) [lenient curve]
- If required < actual ≤ 1.5x: score = 1.0 - 0.05 * (ratio - 1.0)
- If 1.5x < actual ≤ 2.5x: score = 0.95 - 0.10 * (ratio - 1.5)
- If actual > 2.5x: score = 0.85 - 0.10 * (ratio - 2.5), min 0.55

Examples:
- 5y actual, 5y required: 1.0 (perfect)
- 3y actual, 5y required: 0.73 (junior but acceptable)
- 8y actual, 5y required: 0.95 (slightly experienced)
- 12y actual, 5y required: 0.65 (overqualified, may leave)
- 0y actual: 0.50 (fresh graduate baseline)

───────────────────────────────────────────────────────────────────────────

D3: EDUCATION (0.0 - 1.0)
──────────────────────────
Measures: Education level vs requirement

EDUCATION_ORDINAL (Hierarchy):
0: Unknown
1: High School / 12th pass
2: Diploma / Associate
3: Bachelors (B.Tech, B.Sc, B.A, B.Com)
4: Masters (M.Tech, MBA, M.Sc, M.A)
5: Advanced (PhD, MD, LLB, etc.) [≥5 considered "advanced"]

Scoring Logic:
- Candidate level ≥ Required: 1.0 (meets/exceeds)
- Candidate = Required - 1: 0.75 (one level below)
- Candidate = Required - 2: 0.60 (two levels below)
- Candidate < Required - 2: 0.45 (more than 2 levels below)

Special Case: Specialized Degrees
- If JD requires specialized degree (LLB, MBBS, etc.) but candidate
  has different stream (e.g., BE), mark as NOT ELIGIBLE during Step 1
- Regular degree mismatches score via D3 without hard rejection

Examples:
- Masters required, candidate Masters: 1.0
- Masters required, candidate Bachelors: 0.75
- Masters required, candidate Diploma: 0.60
- Masters required, candidate 12th: 0.45

───────────────────────────────────────────────────────────────────────────

D4: AVAILABILITY (0.0 - 1.0)
──────────────────────────────
Measures: How soon candidate can join

Tiers by Notice Period:
- 0 days (immediately): 1.0
- ≤14 days: 0.95
- ≤30 days: 0.90
- ≤60 days: 0.85
- ≤90 days: 0.80
- >90 days: 0.65

Adjustment: If notice period > JD max (default 90), multiply by 0.85
- Example: 120 day notice, JD max 90 → 0.65 * 0.85 = 0.55

Examples:
- Immediately available: 1.0
- 30 days notice, JD allows 90: 0.90
- 120 days notice, JD allows 90: 0.65 * 0.85 = 0.55

───────────────────────────────────────────────────────────────────────────

D5: MISCELLANEOUS (0.0 - 1.0)
───────────────────────────
Measures: Title alignment + overall relevance (soft fit)

Components:
1. Title Similarity (50% weight)
   - SBERT embedding cosine similarity between candidate job titles and JD title
   - Range: 0.0 - 1.0
   - Example: "Backend Engineer" vs "Senior Backend Engineer": ~0.92

2. Resume Relevance (50% weight)
   - SBERT embedding cosine similarity between resume summary and JD description
   - Range: 0.0 - 1.0
   - Example: AI/ML company description vs ML engineer resume: ~0.88

Final Score: max(0.50, 0.5 * title_score + 0.5 * relevance_score)

Floor at 50% ensures:
- Even unrelated background gets minimal credit
- Soft skills and general relevance matter

Examples:
- Perfect title + perfect relevance: 1.0
- Good title + mediocre relevance: 0.75
- Poor title + good relevance: 0.65
- Very poor on both: 0.50

============================================================================
PART 4: ELIGIBILITY RULES (STEP 1)
============================================================================

Philosophy: "No Rejection, Only Positioning"
- Default: ELIGIBLE (only reject in extreme cases)
- Weaker candidates → lower scores, still ranked
- Only mark NOT APPLICABLE (NA) for truly incompatible profiles

Rule Chain (Checked in order):
───────────────────────────────────────────────────────────────────────────

RULE 1: UNRELATED PROFESSION
─────────────────────────────
Trigger: Resume shows completely unrelated professional background

Unrelated Keyword List:
- Yoga: yoga instructor, yoga teacher
- Culinary: chef, sous chef, head chef, pastry chef, cook
- Personal services: hairdresser, beautician, makeup artist, cosmetologist
- Trades: plumber, electrician, carpenter, welder, mason
- Outdoor: gardener, florist, landscaper
- Transport: truck driver, cab driver, delivery driver
- Security: security guard, bouncer
- Hospitality: waiter, waitress, bartender

Detection:
- Strong signal: Keyword in job titles (1+ hit → NA)
- Weak signal: Keyword in resume body (need 3+ hits to trigger)
- Exception: If JD itself is for one of these domains, no filter

Result: NOT APPLICABLE (marked NA)

Example:
- Yoga Teacher → Software Engineer: NA (unrelated)
- Executive Chef → Software Engineer: NA (unrelated)
- Chef → Executive Chef role: ELIGIBLE (matching domain)

───────────────────────────────────────────────────────────────────────────

RULE 2: SKILL OVERLAP
──────────────────────
Trigger: Zero overlap between candidate skills and JD requirements

Logic:
- Extract candidate skills from resume
- Compare with JD required skills
- Apply semantic matching (SBERT, threshold 0.65)
- Implied skills (apriori rules)

Result:
- 0 matches AND candidate has ≥3 skills: NOT APPLICABLE
- Rationale: If they have skills but none overlap AND have no related
  experience, they may be truly incompatible
- Any match or sparse candidate skills: ELIGIBLE

Examples:
- Candidate: ["cooking", "knife skills"], JD needs: ["Python", "Java"] → NA
- Candidate: ["Python", "Docker"], JD needs: ["Python", "AWS"] → ELIGIBLE (Python match)
- Candidate: ["gardening"], JD needs: ["Python"] → ELIGIBLE (not enough to judge)

───────────────────────────────────────────────────────────────────────────

RULE 3: EXPERIENCE FLOOR
─────────────────────────
Trigger: Candidate experience far below requirement

Current Logic: NO HARD REJECTION
- Experience gap handled via D2 scoring (bell curve, floor 50%)
- Even fresh graduates score ≥0.50 on D2

Philosophy: "Let scores speak" — don't pre-filter

Example:
- 0y candidate for 5y role: Eligible, but D2 score = 0.50

───────────────────────────────────────────────────────────────────────────

RULE 4: EDUCATION
──────────────────
Trigger: Specialized degree stream mismatch OR extreme gap

Specialized Degrees (Stream-based filtering):
- Law stream: LLB, LLM
- Medical stream: MBBS, MD, BDS
- Architecture stream: B.Arch
- Arts stream: BFA, Fine Arts
- Education stream: B.Ed

Logic:
- If JD requires specialized stream (e.g., "JD requires LLB")
  AND candidate from different stream (e.g., BE): NOT APPLICABLE
- Different stream = different professional path, not directly transferable

General Degree Gap:
- Gap ≥2 levels (e.g., Diploma vs Masters required): NOT APPLICABLE
- Gap of 1 level (e.g., Bachelors vs Masters): ELIGIBLE (scored via D3)

Examples:
- JD: "LLB required", Candidate: B.Tech (different stream) → NA
- JD: "B.Tech required", Candidate: LLB (different stream) → NA
- JD: "Masters required", Candidate: Diploma (2-level gap) → NA
- JD: "Masters required", Candidate: Bachelors (1-level gap) → ELIGIBLE

============================================================================
PART 5: 6-STEP PIPELINE FLOW
============================================================================

Input: 
- Resume (parsed from PDF/DOCX)
- Job Description (text)

Step 1: ELIGIBILITY CHECK
──────────────────────────
Input: Resume, JD required skills, min experience, required education, JD text
Output: is_eligible (YES/NO), fail_reason, reasoning_trace
Process: Apply 4 eligibility rules (unrelated profession, skill zero-overlap, 
         specialized degree mismatch, education gap ≥2 levels)

If NOT ELIGIBLE:
└─ Mark as NA, stop here, return to user with reason

If ELIGIBLE:
└─ Continue to Step 2

───────────────────────────────────────────────────────────────────────────

Step 2: EXPERT FLAGS
──────────────────
Input: Resume, JD title, experience requirement, education requirement, JD skills
Output: List of 13 flags (BONUS/PENALTY/INFO), reasoning_trace
Process: Forward chaining rule engine, assign flags based on patterns
- Overqualification (experience, education)
- Career quality (gap, stability)
- Alignment (leadership match, exact title)
- Professional development (certifications, upskilling)
- Availability factors (relocation, visa)
- Potential indicators (high potential, availability risk)

Result: List of flags with score modifiers, confidence, reasoning

───────────────────────────────────────────────────────────────────────────

Step 3: 5-DIMENSIONAL SCORING
───────────────────────────────
Input: Resume, JD (skills, requirements), resume flags, SBERT model
Output: 5 dimension scores (0-1), weighted scores, explanations
Process:
1. D1 Technical Skills: keyword + semantic + apriori + relevance
2. D2 Experience: bell curve around requirement
3. D3 Education: ordinal level comparison
4. D4 Availability: notice period tiers
5. D5 Miscellaneous: title similarity + resume relevance (SBERT)

Floor at 0.40-0.50 for each dimension ensures no "zero" scores

───────────────────────────────────────────────────────────────────────────

Step 4: FLAG MODIFIERS (GA WEIGHTS)
────────────────────────────────────
Input: 5 dimension scores, list of flags, dimension map
Output: Adjusted dimension scores
Process:
- For each flag, add score_modifier to target dimension
- E.g., LEADERSHIP_MATCH (+0.08) applied to "Miscellaneous" dimension
- Keep scores in [0, 1] range

Example:
Before: D5 (Misc) = 0.65
Flag: LEADERSHIP_MATCH (+0.08)
After: D5 (Misc) = min(1.0, 0.65 + 0.08) = 0.73

───────────────────────────────────────────────────────────────────────────

Step 5: RANKING (BEST-FIRST SEARCH)
──────────────────────────────────
Input: 5 adjusted dimension scores, GA weights for job category
Output: Overall score (0-100), rank (1, 2, 3, ...)
Process:
1. Get optimized weights from CATEGORY_WEIGHTS (per job category)
2. Calculate weighted sum: overall = Σ(dimension_score × weight) * 100
3. Sort all eligible candidates by overall_score DESC
4. Assign ranks 1, 2, 3, ...
5. Append NA candidates at end with rank -1

Formula: overall_score = (D1×w1 + D2×w2 + D3×w3 + D4×w4 + D5×w5) * 100

Example (Data Science category):
D1=0.85, D2=0.90, D3=0.80, D4=0.85, D5=0.75
Weights: w1=0.2252, w2=0.1833, w3=0.1842, w4=0.2283, w5=0.1790
Overall = (0.85×0.2252 + 0.90×0.1833 + 0.80×0.1842 + 0.85×0.2283 + 0.75×0.1790) * 100
        = (0.1914 + 0.1649 + 0.1474 + 0.1941 + 0.1343) * 100
        = 0.8321 * 100
        = 83.2 / 100

───────────────────────────────────────────────────────────────────────────

Step 6: EXPLAINABLE AI (XAI)
──────────────────────────────
Input: All previous steps' outputs
Output: Reasoning chain, justification, dimension explanations
Process:
- Eligibility trace: Why/why not eligible
- Flags trace: Which flags triggered, reasons
- Dimension explanations: How each score was calculated
- Overall justification: Summary strengths, weaknesses, flags, rank reason

Format: Human-readable narrative explaining every decision

============================================================================
PART 6: WORKING EXAMPLE
============================================================================

CANDIDATE: Sangeeta Bahrani
POSITION: JD_001 — Senior Backend Engineer (ML + Distributed Systems)

STEP 1: ELIGIBILITY CHECK
─────────────────────────
Profession: Lead Test Analyst / QA Lead → ELIGIBLE (tech role, not unrelated)
Skills Overlap: QA Automation + Selenium + Java → Some overlap with backend → ELIGIBLE
Education: Not specified, but let's assume Bachelors → ELIGIBLE
Result: ELIGIBLE ✓

STEP 2: EXPERT FLAGS
────────────────────
- Leadership experience (QA Lead) but not exact backend lead → LEADERSHIP_MISMATCH (-0.02)
- 11 years experience vs 5 required → borderline overqualified → borderline OVERQUALIFICATION
- Java + Selenium skills (relevant to backend indirectly) → UPSKILL_FLAG check: 
  "ISTQB" mentioned → CERTIFICATION_SCORE (+0.03)
- Notice period: Not specified, assume 30 days → RELOCATION_FLAG check: Not mentioned
- Availability Risk: Pursuing MTech AI at NUS → AVAILABILITY_RISK_FLAG (-0.04)
  (1 indicator: pursuing degree)

Summary flags:
├─ CERTIFICATION_SCORE: +0.03
├─ AVAILABILITY_RISK_FLAG: -0.04
└─ (Maybe) LEADERSHIP_MISMATCH: -0.02

STEP 3: 5-DIMENSIONAL SCORING
──────────────────────────────

D1 TECHNICAL SKILLS:
- Resume skills: QA Automation, Selenium, Cucumber, BDD, Java, REST Assured
- JD required: Java, Python, Kafka, Microservices, ML, AWS, API Design, K8s
- Direct matches: Java, REST API (for API Design)
- Semantic matches: Automation experience ≈ CI/CD
- Score: 0.60 (some Java overlap but missing Python, Kafka, ML, K8s)
- Explanation: "QA background has Java + API testing, but lacks ML/distributed systems focus"

D2 EXPERIENCE:
- Actual: 11 years
- Required: 5 years
- Ratio: 2.2x required
- Formula: 0.95 - 0.10 * (2.2 - 1.5) = 0.95 - 0.07 = 0.88
- Score: 0.88
- Explanation: "11y QA experience, well above 5y requirement. Slightly overqualified risk."

D3 EDUCATION:
- Assumed: Bachelors
- Required: Bachelors
- Score: 1.0
- Explanation: "Meets education requirement"

D4 AVAILABILITY:
- Notice: 30 days (assumed, pursuing MTech until May 2026)
- JD max: 90 days
- Tier: ≤30 days → 0.90
- Score: 0.90 - 0.04 (AVAILABILITY_RISK_FLAG penalty applied in Step 4)
- Explanation: "Within 1 month notice, but pursuing MTech until May 2026 may delay start"

D5 MISCELLANEOUS:
- Title: "Lead Test Analyst" vs "Senior Backend Engineer" (different focus)
- SBERT similarity: ~0.55 (similar level, different specialty)
- Resume relevance: QA documentation vs backend systems (moderate overlap)
- Scores: title=0.55, relevance=0.60
- Combined: max(0.50, 0.5×0.55 + 0.5×0.60) = 0.575
- Before flags: 0.58
- Explanation: "Title indicates QA focus, not backend development. Some system testing experience."

STEP 4: APPLY FLAGS & ADJUST SCORES
──────────────────────────────────────

Adjustments:
- D1 (Technical Skills): 0.60 + 0.03 (CERTIFICATION_SCORE) = 0.63
- D2 (Experience): 0.88 (no changes)
- D3 (Education): 1.0 (no changes)
- D4 (Availability): 0.90 - 0.04 (AVAILABILITY_RISK_FLAG) = 0.86
- D5 (Miscellaneous): 0.58 - 0.02 (LEADERSHIP_MISMATCH) = 0.56

STEP 5: CALCULATE OVERALL SCORE
──────────────────────────────────

Category: Data Science (detected from JD)
Weights: w_skills=0.2252, w_exp=0.1833, w_edu=0.1842, w_avail=0.2283, w_misc=0.1790

Overall = (0.63×0.2252 + 0.88×0.1833 + 1.0×0.1842 + 0.86×0.2283 + 0.56×0.1790) × 100
        = (0.1419 + 0.1613 + 0.1842 + 0.1963 + 0.1002) × 100
        = 0.7839 × 100
        = 78.4 / 100

Rank: 9th out of 10 eligible candidates for this JD (from ground truth)

STEP 6: GENERATE JUSTIFICATION
─────────────────────────────

"Sangeeta Bahrani scores 78.4/100. Strengths: Education (100%), Availability (86%), 
Experience (88%). Areas to develop: Technical Skills (63%) — lacks ML/distributed 
systems depth; Miscellaneous (56%) — background is QA, not backend. Flags: 
CERTIFICATION_SCORE (ISTQB), AVAILABILITY_RISK_FLAG (pursuing MTech, may impact 
start date), possible overqualification. Verdict: Ranked 9/10. Strong on experience 
and availability, but significant skill gap in core backend/ML technologies. Suitable 
for intermediate or QA lead role, not ideal for Senior Backend Engineer position."

============================================================================
PART 7: QUICK REFERENCE — FLAG PENALTIES/BONUSES
============================================================================

PENALTY FLAGS (Negative Impact on Scores):
┌─────────────────────────────────────────────────────────────────────────┐
│ Flag Name                    │ Penalty │ Dimension      │ Condition    │
├─────────────────────────────────────────────────────────────────────────┤
│ OVERQUALIFICATION_FLAG       │ -6%     │ Experience     │ 2.5x+ exp    │
│ OVERQUALIFICATION_EDU        │ -4%     │ Education      │ Advanced edu │
│ LEADERSHIP_MISMATCH          │ -2%     │ Miscellaneous  │ Lead for IC  │
│ CAREER_GAP_FLAG              │ -4%     │ Miscellaneous  │ >12mo gap    │
│ CAREER_STABILITY_FLAG        │ -5%     │ Miscellaneous  │ 4+ roles/5y  │
│ EDUCATION_GAP_FLAG           │ -3%/lvl │ Education      │ <Required    │
│ WORK_VISA_FLAG               │ -3%     │ Availability   │ Visa needed  │
│ AVAILABILITY_RISK_FLAG       │ -4%     │ Availability   │ 2+ risk ind. │
└─────────────────────────────────────────────────────────────────────────┘

BONUS FLAGS (Positive Impact on Scores):
┌─────────────────────────────────────────────────────────────────────────┐
│ Flag Name                    │ Bonus   │ Dimension      │ Condition    │
├─────────────────────────────────────────────────────────────────────────┤
│ LEADERSHIP_MATCH             │ +8%     │ Miscellaneous  │ Lead for Sr  │
│ EXACT_TITLE_MATCH            │ +7%     │ Miscellaneous  │ Exact title  │
│ CERTIFICATION_SCORE (1 cert) │ +3%     │ Technical      │ Cert present │
│ CERTIFICATION_SCORE (3 certs)│ +9%     │ Technical      │ 3+ certs     │
│ UPSKILL_FLAG                 │ +3%     │ Technical      │ Learning ev. │
│ RELOCATION_FLAG              │ +3%     │ Availability   │ Open to move │
│ WORK_VISA_FLAG (Positive)    │ +2%     │ Availability   │ JD sponsors  │
│ HIGH_POTENTIAL_FLAG          │ +5%     │ Miscellaneous  │ 2+ indicators│
└─────────────────────────────────────────────────────────────────────────┘

============================================================================
"""
