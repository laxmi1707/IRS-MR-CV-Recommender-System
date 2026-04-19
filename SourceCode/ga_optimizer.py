"""
ga_optimizer.py — Step 5: Genetic Algorithm Weight Optimization
Offline Calibration via DEAP

Evolves optimal dimension weights per job category.
- Chromosome: [w_skills, w_experience, w_education, w_availability, w_misc]
- Constraint: sum(weights) = 1.0
- Fitness: Kendall Tau rank correlation against ground truth
- Selection: Tournament (k=3)
- Crossover: BLX-α (α=0.5), cxpb=0.7
- Mutation: Gaussian (σ=0.1), mutpb=0.2
- Termination: 50 generations + early stop (10 stall)
"""

import random
import math
import numpy as np
from scipy.stats import kendalltau


# ─── Pre-optimized Weights by Job Category ────────────────────
# These are the "offline" results — in production, GA runs once
# per job category on training data and stores the optimal weights.
# The runtime system looks up these cached weights.

CATEGORY_WEIGHTS = {
    "data_science": {
        "technical_skills": 0.40,
        "experience": 0.30,
        "education": 0.12,
        "availability": 0.08,
        "miscellaneous": 0.10,
    },
    "software_engineering": {
        "technical_skills": 0.38,
        "experience": 0.28,
        "education": 0.10,
        "availability": 0.10,
        "miscellaneous": 0.14,
    },
    "contract": {
        "technical_skills": 0.25,
        "experience": 0.15,
        "education": 0.08,
        "availability": 0.39,
        "miscellaneous": 0.13,
    },
    "finance": {
        "technical_skills": 0.30,
        "experience": 0.20,
        "education": 0.25,
        "availability": 0.10,
        "miscellaneous": 0.15,
    },
    "management": {
        "technical_skills": 0.20,
        "experience": 0.35,
        "education": 0.10,
        "availability": 0.10,
        "miscellaneous": 0.25,
    },
    "default": {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.20,
        "availability": 0.10,
        "miscellaneous": 0.10,
    },
}

# ─── Job Category Detection ──────────────────────────────────
def detect_job_category(jd_title: str, jd_text: str) -> str:
    """Infer job category from JD for weight lookup."""
    combined = (jd_title + " " + jd_text).lower()

    if any(kw in combined for kw in [
        "data science", "machine learning", "ml engineer", "data analyst",
        "analytics", "deep learning", "ai engineer",
    ]):
        return "data_science"

    if any(kw in combined for kw in [
        "software engineer", "developer", "backend", "frontend",
        "full stack", "devops", "sre", "platform engineer","quality assurance", "qa", "test engineer","automation engineer",
        "mobile developer", "ios developer", "android developer","technical analyst", "systems engineer", "cloud engineer",
    ]):
        return "software_engineering"

    if any(kw in combined for kw in [
        "contract", "freelance", "temporary", "6 month", "12 month",
        "fixed term", "interim",
    ]):
        return "contract"

    if any(kw in combined for kw in [
        "finance", "accounting", "banking", "audit", "risk",
        "compliance", "investment", "actuary","swift", "fintech", "quantitative analyst", "quant", "financial analyst", "treasury", "controller", "cfo",
        "foreign exchange", "fx", "hedge fund", "private equity", "venture capital",
    ]):
        return "finance"

    if any(kw in combined for kw in [
        "manager", "director", "head of", "vp", "chief",
        "leadership", "executive","technical manager", "project manager", "product manager", "program manager", "operations manager", "general manager",
    ]):
        return "management"

    return "default"


def get_optimized_weights(jd_title: str, jd_text: str) -> tuple[dict[str, float], str]:
    """
    Get GA-optimized weights for the detected job category.
    In production, these would be loaded from a pre-trained GA run.
    """
    category = detect_job_category(jd_title, jd_text)
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["default"])
    return weights , category


# ═══════════════════════════════════════════════════════════════
# GA ENGINE (for offline training — not called at runtime)
# Kept here for completeness and demonstration
# ═══════════════════════════════════════════════════════════════

def normalize_weights(individual):
    """Ensure weights sum to 1.0."""
    total = sum(individual)
    if total == 0:
        individual[:] = [0.35, 0.25, 0.20, 0.10, 0.10]
    else:
        individual[:] = [w / total for w in individual]
    return individual


def evaluate_fitness(individual, candidate_scores, ground_truth_ranks):
    """
    Fitness function: Kendall Tau rank correlation.

    Args:
        individual: [w1, w2, w3, w4, w5] weight chromosome
        candidate_scores: list of dicts with dim scores per candidate
        ground_truth_ranks: expected ranking order

    Returns:
        (kendall_tau,) — higher is better
    """
    individual = normalize_weights(list(individual))
    dim_keys = ["technical_skills", "experience", "education",
                "availability", "miscellaneous"]

    # Compute weighted scores
    predicted_scores = []
    for candidate in candidate_scores:
        score = sum(
            candidate[dim_keys[i]] * individual[i]
            for i in range(5)
        )
        predicted_scores.append(score)

    # Compute Kendall Tau correlation
    
    predicted_ranks = np.argsort(np.argsort([-s for s in predicted_scores]))
    tau, _ = kendalltau(predicted_ranks, np.array(ground_truth_ranks))

    if tau is None or math.isnan(tau):
        tau = 0.0
    return (tau,)

def run_ga_optimization(
    candidate_scores: list[dict],
    ground_truth_ranks: list[int],
    n_generations: int = 50,
    population_size: int = 50,
    early_stop_patience: int = 10,
) -> dict[str, float]:
    """
    Run the full GA optimization.

    This is the OFFLINE phase — runs once per job category
    on labelled training data to discover optimal weights.

    Args:
        candidate_scores: list of {dim_name: score} per candidate
        ground_truth_ranks: expert-labeled correct ranking
        n_generations: max generations
        population_size: chromosomes per generation
        early_stop_patience: stop if no improvement for N gens

    Returns:
        dict of optimized weights
    """
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = [random.uniform(0.05, 0.5) for _ in range(5)]
        normalize_weights(individual)
        population.append(individual)

    best_fitness = -1.0
    stall_count = 0
    best_individual = population[0]

    dim_keys = ["technical_skills", "experience", "education",
                "availability", "miscellaneous"]

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = [
            evaluate_fitness(ind, candidate_scores, ground_truth_ranks)[0]
            for ind in population
        ]

        # Track best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = float(fitnesses[gen_best_idx])

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx][:]
            stall_count = 0
        else:
            stall_count += 1

        # Early stopping
        if stall_count >= early_stop_patience:
            break

        # Selection: Tournament (k=3)
        selected = []
        for _ in range(population_size):
            tournament = random.sample(range(population_size), k=3)
            winner = max(tournament, key=lambda i: fitnesses[i])
            selected.append(population[winner][:])

        # Crossover: BLX-α (α=0.5)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            if random.random() < 0.7:  # cxpb
                child1, child2 = [], []
                for j in range(5):
                    alpha = 0.5
                    d = abs(p1[j] - p2[j])
                    low = min(p1[j], p2[j]) - alpha * d
                    high = max(p1[j], p2[j]) + alpha * d
                    child1.append(max(0.01, random.uniform(low, high)))
                    child2.append(max(0.01, random.uniform(low, high)))
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1[:], p2[:]])

        # Mutation: Gaussian (σ=0.1)
        for ind in offspring:
            if random.random() < 0.2:  # mutpb
                idx = random.randint(0, 4)
                ind[idx] += random.gauss(0, 0.1)
                ind[idx] = max(0.01, ind[idx])

        # Normalize all offspring
        for ind in offspring:
            normalize_weights(ind)

        population = offspring[:population_size]

    # Return best weights
    normalize_weights(best_individual)
    return {
        dim_keys[i]: round(best_individual[i], 4)
        for i in range(5)
    }
