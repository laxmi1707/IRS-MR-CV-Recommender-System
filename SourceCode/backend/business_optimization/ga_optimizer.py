"""
ga_optimizer.py — Step 5: Genetic Algorithm Weight Optimization
Offline Calibration via DEAP-style GA

Chromosome: [w_skills, w_experience, w_education, w_availability, w_misc]
Fitness: Kendall Tau rank correlation
Selection: Tournament (k=5), Crossover: BLX-α (0.5), Mutation: Gaussian (σ=0.1)
Termination: 50 generations + early stop (10 stall)
"""

import random
import math
import numpy as np
from scipy.stats import kendalltau


# Pre-optimized weights by job category (trained on ground truth data)
# Calibrated using GA optimization against JDVsCDRanking.csv
CATEGORY_WEIGHTS = {
    "data_science": {
        "technical_skills": 0.2252,
        "experience": 0.1833,
        "education": 0.1842,
        "availability": 0.2283,
        "miscellaneous": 0.1790,
    },
    "software_engineering": {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.20,
        "availability": 0.10,
        "miscellaneous": 0.10,
    },
    "contract": {
        "technical_skills": 0.25,
        "experience": 0.15,
        "education": 0.08,
        "availability": 0.39,
        "miscellaneous": 0.13,
    },
    "finance": {
        "technical_skills": 0.1645,
        "experience": 0.3234,
        "education": 0.1379,
        "availability": 0.2455,
        "miscellaneous": 0.1288,
    },
    "management": {
        "technical_skills": 0.1350,
        "experience": 0.2091,
        "education": 0.2102,
        "availability": 0.1369,
        "miscellaneous": 0.3088,
    },
    "entry_level": {
        "technical_skills": 0.1557,
        "experience": 0.1564,
        "education": 0.2190,
        "availability": 0.2304,
        "miscellaneous": 0.2386,
    },
    "default": {
        "technical_skills": 0.1852,
        "experience": 0.2538,
        "education": 0.1152,
        "availability": 0.2618,
        "miscellaneous": 0.1839,
    },
}

DEFAULT_WEIGHTS_LIST = [0.35, 0.25, 0.20, 0.10, 0.10]


def detect_job_category(jd_title: str, jd_text: str) -> str:
    """Infer job category from JD for weight lookup."""
    combined = (jd_title + " " + jd_text).lower()

    if any(kw in combined for kw in [
        "contract", "freelance", "temporary", "6 month", "12 month",
        "fixed term", "interim",
    ]):
        return "contract"

    if any(kw in combined for kw in [
        "entry level", "entry-level", "fresh graduate", "fresher",
        "junior", "associate", "trainee", "intern",
    ]) and "senior" not in combined:
        return "entry_level"

    if any(kw in combined for kw in [
        "head of", "director", "vp ", "vice president", "chief",
        "cto", "ceo", "cfo", "coo", "executive",
        "program manager", "general manager", "operations manager",
    ]):
        return "management"

    if any(kw in combined for kw in [
        "data scien", "machine learning", "ml engineer", "data analyst",
        "analytics", "deep learning", "ai engineer", "data engineer",
        "nlp", "computer vision",
    ]):
        return "data_science"

    if any(kw in combined for kw in [
        "software engineer", "developer", "backend", "frontend",
        "full stack", "full-stack", "devops", "sre", "platform engineer",
        "quality assurance", "qa ", "test engineer", "automation engineer",
        "mobile developer", "ios developer", "android developer",
        "technical analyst", "systems engineer", "cloud engineer",
        "web developer", "application developer", "site reliability",
    ]):
        return "software_engineering"

    if any(kw in combined for kw in [
        "finance", "accounting", "banking", "audit", "risk analyst",
        "compliance", "investment", "actuary", "fintech",
        "quantitative analyst", "financial analyst", "treasury",
        "controller", "foreign exchange", "hedge fund",
    ]):
        return "finance"

    return "default"


def get_optimized_weights(jd_title: str, jd_text: str):
    """Get GA-optimized weights for detected job category."""
    category = detect_job_category(jd_title, jd_text)
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["default"])
    return weights, category


# ═══════════════════════════════════════════════════════════════
# GA ENGINE (offline training)
# ═══════════════════════════════════════════════════════════════

def normalize_weights(individual):
    """Ensure weights sum to 1.0."""
    total = sum(individual)
    if total == 0:
        individual[:] = DEFAULT_WEIGHTS_LIST[:]
    else:
        individual[:] = [w / total for w in individual]
    return individual


def evaluate_fitness(individual, candidate_scores, ground_truth_ranks):
    """Fitness: Kendall Tau rank correlation. Higher = better."""
    individual = normalize_weights(list(individual))
    dim_keys = ["technical_skills", "experience", "education",
                "availability", "miscellaneous"]

    predicted_scores = []
    for candidate in candidate_scores:
        score = sum(candidate[dim_keys[i]] * individual[i] for i in range(5))
        predicted_scores.append(score)

    predicted_ranks = np.argsort(np.argsort([-s for s in predicted_scores]))

    tau = 0.0
    try:
        result = kendalltau(predicted_ranks, np.array(ground_truth_ranks))
        tau_val = float(result[0])
        if not math.isnan(tau_val):
            tau = tau_val
    except Exception:
        tau = 0.0

    return (tau,)


def run_ga_optimization(
    candidate_scores,
    ground_truth_ranks,
    n_generations: int = 50,
    population_size: int = 50,
    early_stop_patience: int = 10,
):
    """Full GA optimization. Offline phase — once per job category."""
    dim_keys = ["technical_skills", "experience", "education",
                "availability", "miscellaneous"]

    population = []
    for _ in range(population_size):
        individual = [random.uniform(0.05, 0.5) for _ in range(5)]
        normalize_weights(individual)
        population.append(individual)

    best_fitness = -1.0
    stall_count = 0
    best_individual = population[0][:]

    for gen in range(n_generations):
        fitnesses = [
            evaluate_fitness(ind, candidate_scores, ground_truth_ranks)[0]
            for ind in population
        ]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = float(fitnesses[gen_best_idx])

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx][:]
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= early_stop_patience:
            break

        # Tournament selection (k=3)
        selected = []
        for _ in range(population_size):
            tournament = random.sample(range(population_size), k=3)
            winner = max(tournament, key=lambda i: fitnesses[i])
            selected.append(population[winner][:])

        # BLX-α crossover (α=0.5)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            if random.random() < 0.7:
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

        # Gaussian mutation (σ=0.1)
        for ind in offspring:
            if random.random() < 0.2:
                idx = random.randint(0, 4)
                ind[idx] += random.gauss(0, 0.1)
                ind[idx] = max(0.01, ind[idx])

        for ind in offspring:
            normalize_weights(ind)

        population = offspring[:population_size]

    normalize_weights(best_individual)
    return {dim_keys[i]: round(best_individual[i], 4) for i in range(5)}
