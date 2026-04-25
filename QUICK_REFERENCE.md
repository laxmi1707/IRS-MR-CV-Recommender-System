# Quick Reference Guide — GA Optimizer Implementation

## Files Modified/Created

```
SourceCode/
├── train_ga_weights.py          [NEW] GA training script
├── ga_optimizer.py              [UPDATED] New weights, detect_job_category, get_optimized_weights
├── expert_flags.py              [UPDATED] Added HIGH_POTENTIAL_FLAG, AVAILABILITY_RISK_FLAG
├── eligibility_engine.py         [REVIEWED] No changes needed
├── scoring_engine.py            [REVIEWED] No changes needed
├── resume_parser.py             [NO CHANGES]
├── main.py                      [NO CHANGES]
└── requirements.txt             [CHECK] Ensure pandas, openpyxl installed

Root/
├── IMPLEMENTATION_GUIDE.md      [NEW] Comprehensive reference (7 parts)
├── CHANGES_SUMMARY.md           [NEW] Change log & deployment checklist
└── README.md                    [EXISTING] Update if needed
```

## Quick Start

### 1. Train GA Weights (One-time offline)
```bash
cd SourceCode
python train_ga_weights.py
```

Output shows:
- Per-JD optimization results
- Aggregated weights by category
- Python code to update ga_optimizer.py (already done)

### 2. Run the API
```bash
cd SourceCode
python -m uvicorn main:app --reload
```

Default: http://localhost:8000

Health check: http://localhost:8000/health

### 3. Test Endpoint
```bash
POST http://localhost:8000/api/rank
Parameters:
- job_title: "Software Engineer"
- job_description: "We need a Python developer..."
- resumes: [File1.pdf, File2.pdf, ...]
- weights (optional): JSON custom weights

Response:
{
  "success": true,
  "candidates": [
    {
      "rank": 1,
      "name": "John Doe",
      "overall_score": 85.3,
      "is_eligible": true,
      "dimensions": [...],
      "expert_flags": [...],
      "reasoning_chain": "..."
    }
  ]
}
```

## Penalty & Bonus Quick Reference

### Penalties (Negative Impact)
| Flag | Amount | When |
|------|--------|------|
| OVERQUALIFICATION | -6% | 2.5x+ experience |
| EDUCATION_GAP | -3%/level | Below requirement |
| CAREER_STABILITY | -5% | 4+ jobs in 5y |
| CAREER_GAP | -4% | >12mo break |
| AVAILABILITY_RISK | -4% | 2+ risk factors |
| WORK_VISA | -3% | Needs visa, no offer |
| OVERQUALIFICATION_EDU | -4% | Advanced edu/junior role |
| LEADERSHIP_MISMATCH | -2% | Lead for IC role |

### Bonuses (Positive Impact)
| Flag | Amount | When |
|------|--------|------|
| LEADERSHIP_MATCH | +8% | Lead for lead role |
| EXACT_TITLE_MATCH | +7% | Same job title |
| CERTIFICATION (3x) | +9% | 3+ certifications |
| CERTIFICATION (1x) | +3% | 1 certification |
| HIGH_POTENTIAL | +5% | 2+ growth indicators |
| RELOCATION | +3% | Open to move |
| UPSKILL | +3% | Continuous learning |
| WORK_VISA (positive) | +2% | JD sponsors visa |

## 5-Dimensional Scores Overview

| Dimension | Method | Range | Floor | When High |
|-----------|--------|-------|-------|-----------|
| D1: Technical Skills | Keyword + semantic + apriori + relevance | 0-1 | 0.40 | Skills match JD needs |
| D2: Experience | Bell curve around requirement | 0-1 | 0.50 | 0.8x-1.5x required |
| D3: Education | Ordinal level (1-5 scale) | 0-1 | 0.45 | Meets/exceeds level |
| D4: Availability | Notice period tiers | 0-1 | 0.65 | Immediate/quick start |
| D5: Miscellaneous | Title + resume similarity (SBERT) | 0-1 | 0.50 | Title & relevance match |

## Optimized Weights by Category

### Data Science (skills + availability emphasized)
- Skills: 22.5% | Experience: 18.3% | Education: 18.4% | Availability: 22.8% | Misc: 17.9%

### Management (soft skills paramount)
- Skills: 13.5% | Experience: 20.9% | Education: 21.0% | Availability: 13.7% | **Misc: 30.9%**

### Finance (experience critical)
- Skills: 16.5% | **Experience: 32.3%** | Education: 13.8% | Availability: 24.6% | Misc: 12.9%

### Entry-Level (education + soft skills)
- Skills: 15.6% | Experience: 15.6% | **Education: 21.9%** | **Availability: 23.0%** | **Misc: 23.9%**

### Default (experience + availability)
- Skills: 18.5% | **Experience: 25.4%** | Education: 11.5% | **Availability: 26.2%** | Misc: 18.4%

## Eligibility Rules (Step 1)

Candidate is NOT APPLICABLE if:
1. **Unrelated profession**: Yoga teacher, Chef, Plumber, etc. for tech roles
2. **Zero skill overlap**: No matches AND candidate has ≥3 skills
3. **Specialized degree mismatch**: LLB vs B.Tech when JD requires specific stream
4. **Education gap ≥2 levels**: Diploma when Masters required

Otherwise: **ELIGIBLE** (proceeds to scoring)

## 6-Step Pipeline

1. **Eligibility Check** → is_eligible (YES/NO)
   - If NO → mark as NA, stop
   - If YES → continue

2. **Expert Flags** → 13 flags detected (BONUS/PENALTY/INFO)
   - Calculate all applicable flags
   - Track confidence levels

3. **5-Dimensional Scoring** → 5 scores (D1-D5)
   - Each in range [0, 1]
   - Each floored to prevent harsh zeros

4. **Apply Flag Modifiers** → Adjusted scores
   - Add bonus/penalty to target dimension
   - Keep scores in [0, 1]

5. **Ranking** → Overall score (0-100)
   - Formula: (D1×w1 + D2×w2 + ... + D5×w5) × 100
   - Weights from CATEGORY_WEIGHTS
   - Sort DESC, assign ranks 1, 2, 3, ...

6. **XAI Explanation** → Reasoning chain
   - Why eligible/ineligible
   - Which flags triggered
   - How each dimension was scored
   - Why ranked in that position

## Ground Truth Data

Source: `dataset/GroundTruth/GroundTruth_Data_Candidate Ranking with JD Id.xlsx`

Structure:
- 120 candidates across 8 job descriptions
- 63 ELIGIBLE (52.5%), 57 NOT APPLICABLE (47.5%)
- 5-dimensional expert scores per candidate (0-100 scale)
- Expert rankings (1, 2, 3, ... per JD)
- Additional flags: Relocation, Work Visa, Leadership, High Potential, Overqualification, Availability Risk

Training process normalized scores to 0-1 and optimized weights using GA.

## Common Issues & Solutions

### Issue: Weights not updating
**Solution**: Ensure ga_optimizer.py has latest CATEGORY_WEIGHTS from train_ga_weights.py output

### Issue: Flags not detected
**Solution**: Check resume text format, ensure keywords are present, review flag trigger conditions

### Issue: Scores too high/low
**Solution**: Review dimension-specific logic in scoring_engine.py, check floor values

### Issue: Ranking differs from expected
**Solution**: Verify GA category detection, check weights applied, review flag modifiers

### Issue: SBERT model fails to load
**Solution**: Ensure `sentence-transformers` package installed, check internet connection for model download

## Performance Targets

- Eligibility filter accuracy: 95%+
- Flag detection accuracy: 85%+
- Rank correlation with expert (Kendall Tau): ≥0.7
- Average overall score: 60-70
- Score std deviation: 15-20

## Documentation Files

1. **IMPLEMENTATION_GUIDE.md** - Comprehensive 7-part reference
   - GA optimization results
   - All 13 flags with penalties/bonuses
   - 5-dimensional scoring definitions
   - Eligibility rules
   - 6-step pipeline
   - Working example walkthrough
   - Quick reference tables

2. **CHANGES_SUMMARY.md** - What changed and why
   - New/updated files
   - Specific changes per file
   - Verification checklist
   - Deployment guide
   - Future improvements

3. **This file** - Quick reference for common tasks

## Next Steps

1. ✅ Verify train_ga_weights.py runs without errors
2. ✅ Check ga_optimizer.py has updated CATEGORY_WEIGHTS
3. ✅ Review expert_flags.py for new flags
4. ✅ Test end-to-end pipeline with sample data
5. ✅ Spot-check against ground truth (5-10 candidates)
6. 🔄 Deploy to production
7. 🔄 Monitor and collect feedback
8. 🔄 Plan quarterly weight retraining

## Key Improvements Made

✅ **GA Optimization**: Weights now calibrated to ground truth rankings
✅ **Expert Flags**: Added HIGH_POTENTIAL and AVAILABILITY_RISK flags
✅ **Reproducibility**: Created train_ga_weights.py for offline training
✅ **Documentation**: Comprehensive guides for implementation and usage
✅ **Validation**: Verified against 120 ground truth candidates

## Support

For detailed information:
- GA weights: See IMPLEMENTATION_GUIDE.md Part 1
- Expert flags: See IMPLEMENTATION_GUIDE.md Part 2
- Scoring system: See IMPLEMENTATION_GUIDE.md Parts 3-5
- Eligibility: See IMPLEMENTATION_GUIDE.md Part 4
- Pipeline: See IMPLEMENTATION_GUIDE.md Parts 5-6
- Example: See IMPLEMENTATION_GUIDE.md Part 6
