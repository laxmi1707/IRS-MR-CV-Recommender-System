"""
train_ga_weights.py — GA Optimization Training
Offline weight calibration using ground truth data from Excel

Reads:
- GroundTruth_Data_Candidate Ranking with JD Id.xlsx
- JD_Master_List.xlsx

Outputs:
- Optimized weights per job category
- Performance metrics (Kendall Tau scores)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os

SOURCE_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SOURCE_CODE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CODE_DIR)

from business_optimization.ga_optimizer import run_ga_optimization, detect_job_category

# File paths
GT_FILE = "../dataset/GroundTruth/GroundTruth_Data_Candidate Ranking with JD Id.xlsx"
JD_FILE = "../dataset/GroundTruth/JD_Master_List.xlsx"


def load_ground_truth_data():
    """Load and parse ground truth Excel files."""
    print("[LOAD] Reading ground truth data...")
    
    # Load candidate rankings
    gt_df = pd.read_excel(GT_FILE)
    
    # Load job descriptions
    jd_df = pd.read_excel(JD_FILE)
    
    print(f"[LOAD] Ground truth shape: {gt_df.shape}")
    print(f"[LOAD] JD Master List shape: {jd_df.shape}")
    
    return gt_df, jd_df


def create_jd_mapping(jd_df):
    """Create a mapping of JD_ID to full job description."""
    jd_map = {}
    for _, row in jd_df.iterrows():
        jd_id = row['JD_ID']
        description = row['Full Job Description']
        jd_map[jd_id] = description
    return jd_map


def extract_training_data(gt_df, jd_map):
    """
    Extract training data per JD.
    
    Returns:
        dict: {JD_ID: {"candidate_scores": [...], "ground_truth_ranks": [...], "jd_text": ...}}
    """
    training_data = defaultdict(lambda: {"candidate_scores": [], "ground_truth_ranks": [], "jd_text": ""})
    
    # Group by JD_ID
    for jd_id, jd_group in gt_df.groupby('JD_ID'):
        # Filter only eligible candidates (as per ground truth)
        eligible_candidates = jd_group[jd_group['Eligible'] == 'YES'].copy()
        
        if len(eligible_candidates) == 0:
            print(f"[WARNING] JD '{jd_id}' has 0 eligible candidates. Skipping.")
            continue
        
        # Sort by rank to maintain order
        eligible_candidates = eligible_candidates.sort_values('Final_Rank_Per_JD')
        
        # Extract 5-dimensional scores (normalized to 0-1)
        candidate_scores = []
        ground_truth_ranks = []
        
        for idx, (rank, row) in enumerate(eligible_candidates[['Final_Rank_Per_JD', 'skills_score', 
                                                                   'experience_score', 'education_score',
                                                                   'availability_score', 'misc_score']].iterrows()):
            scores_dict = {
                "technical_skills": row['skills_score'] / 100.0,
                "experience": row['experience_score'] / 100.0,
                "education": row['education_score'] / 100.0,
                "availability": row['availability_score'] / 100.0,
                "miscellaneous": row['misc_score'] / 100.0,
            }
            candidate_scores.append(scores_dict)
            
            # Convert rank to 0-based index (1 → 0, 2 → 1, etc.)
            ground_truth_ranks.append(int(row['Final_Rank_Per_JD']) - 1)
        
        # Get JD text
        jd_text = jd_map.get(jd_id, "")
        
        training_data[jd_id] = {
            "candidate_scores": candidate_scores,
            "ground_truth_ranks": ground_truth_ranks,
            "jd_text": jd_text,
            "num_candidates": len(candidate_scores),
        }
    
    return training_data


def run_optimization_per_jd(training_data):
    """
    Run GA optimization for each JD.
    
    Returns:
        dict: {JD_ID: {"optimized_weights": {...}, "fitness": float, "category": str}}
    """
    results = {}
    
    for jd_id, data in sorted(training_data.items()):
        print(f"\n[GA] Processing JD: {jd_id}")
        print(f"[GA]   - Candidates: {data['num_candidates']}")
        print(f"[GA]   - Running GA optimization...")
        
        # Detect category
        jd_title = jd_id.split(" — ")[1] if " — " in jd_id else jd_id
        category = detect_job_category(jd_title, data['jd_text'][:500])
        print(f"[GA]   - Detected category: {category}")
        
        # Run GA
        try:
            optimized_weights = run_ga_optimization(
                candidate_scores=data['candidate_scores'],
                ground_truth_ranks=data['ground_truth_ranks'],
                n_generations=100,  # More iterations for better convergence
                population_size=100,
                early_stop_patience=15,
            )
            
            results[jd_id] = {
                "optimized_weights": optimized_weights,
                "category": category,
                "num_candidates": data['num_candidates'],
                "status": "SUCCESS"
            }
            
            print(f"[GA]   ✓ Optimized weights:")
            for dim, weight in sorted(optimized_weights.items()):
                print(f"[GA]     {dim}: {weight:.4f}")
        
        except Exception as e:
            print(f"[GA]   ✗ ERROR: {str(e)}")
            results[jd_id] = {
                "optimized_weights": {},
                "category": category,
                "num_candidates": data['num_candidates'],
                "status": f"FAILED: {str(e)}"
            }
    
    return results


def aggregate_by_category(results):
    """
    Aggregate optimized weights by job category.
    
    Returns:
        dict: {category: {"weights": {...}, "jd_count": int, "avg_fitness": float}}
    """
    category_weights = defaultdict(lambda: {"weights_list": [], "jd_count": 0})
    
    for jd_id, result in results.items():
        if result['status'] == "SUCCESS":
            category = result['category']
            category_weights[category]['weights_list'].append(result['optimized_weights'])
            category_weights[category]['jd_count'] += 1
    
    # Average weights per category
    aggregated = {}
    for category, data in category_weights.items():
        if data['weights_list']:
            avg_weights = {}
            dim_keys = ["technical_skills", "experience", "education", "availability", "miscellaneous"]
            
            for dim in dim_keys:
                values = [w[dim] for w in data['weights_list'] if dim in w]
                if values:
                    avg_weights[dim] = round(np.mean(values), 4)
            
            # Normalize to sum to 1.0
            total = sum(avg_weights.values())
            if total > 0:
                avg_weights = {k: round(v / total, 4) for k, v in avg_weights.items()}
            
            aggregated[category] = {
                "weights": avg_weights,
                "jd_count": data['jd_count'],
            }
    
    return aggregated


def save_results(results, aggregated):
    """Save results to CSV and display summary."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    print("\nPer-JD Results:")
    print("-" * 80)
    for jd_id, result in sorted(results.items()):
        status = result['status']
        category = result['category']
        num_cand = result['num_candidates']
        print(f"{jd_id[:50]:<50} | Category: {category:<20} | Candidates: {num_cand:<3} | {status}")
    
    print("\nAggregated Weights by Category:")
    print("-" * 80)
    for category, data in sorted(aggregated.items()):
        print(f"\n{category.upper()}")
        print(f"  JD Count: {data['jd_count']}")
        print(f"  Weights:")
        for dim, weight in sorted(data['weights'].items()):
            print(f"    {dim:<20}: {weight:.4f}")


def generate_python_code(aggregated):
    """Generate Python code for CATEGORY_WEIGHTS dictionary."""
    print("\n" + "=" * 80)
    print("PYTHON CODE FOR ga_optimizer.py")
    print("=" * 80)
    print("\nCATEGORY_WEIGHTS = {")
    for category, data in sorted(aggregated.items()):
        print(f'    "{category}": {{')
        for dim, weight in sorted(data['weights'].items()):
            print(f'        "{dim}": {weight},')
        print(f"    }},")
    print("}")
    

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("GA WEIGHT OPTIMIZATION TRAINING")
    print("=" * 80)
    
    # Load data
    gt_df, jd_df = load_ground_truth_data()
    jd_map = create_jd_mapping(jd_df)
    
    # Extract training data
    print("\n[EXTRACT] Extracting training data per JD...")
    training_data = extract_training_data(gt_df, jd_map)
    print(f"[EXTRACT] Total JDs with eligible candidates: {len(training_data)}")
    for jd_id, data in training_data.items():
        print(f"  - {jd_id[:60]:<60}: {data['num_candidates']} candidates")
    
    # Run GA optimization
    print("\n[OPTIMIZE] Starting GA optimization for each JD...")
    results = run_optimization_per_jd(training_data)
    
    # Aggregate by category
    print("\n[AGGREGATE] Aggregating weights by job category...")
    aggregated = aggregate_by_category(results)
    
    # Save results
    save_results(results, aggregated)
    generate_python_code(aggregated)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
