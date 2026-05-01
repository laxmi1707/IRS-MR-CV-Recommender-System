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
        "miscellaneous": 0.12,
        "availability": 0.08,
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


def detect_job_category(jd_title: str, jd_text: str, sbert_model=None) -> str:
    """Infer job category from JD for weight lookup.

    Two-pass approach for accuracy:
      1. Keyword scoring with TITLE weighted 3x over body text. The title is a
         very strong signal; body text is noisy ('management' often appears in
         developer JDs as 'team management' or 'task management', and that
         shouldn't flip the category to 'management').
      2. SBERT semantic fallback when keyword scoring is ambiguous (top-2 scores
         are tied or both very low). Compares the JD title to canonical
         category descriptions and picks the closest match.

    The previous implementation used `any(kw in combined for kw in [...])` —
    a single substring hit anywhere in the JD flipped the category. That's
    why 'Software Test Engineer' was being detected as 'management' (because
    body text mentioned 'manage' or 'management'). The new scoring system
    requires the title to corroborate the category.
    """
    title_lower = (jd_title or "").lower().strip()
    body_lower = (jd_text or "").lower()

    # Category keyword bundles — same content as before, but now scored
    # rather than first-match-wins.
    CATEGORY_KEYWORDS = {
        "contract": [
            "contract", "freelance", "temporary", "6 month", "12 month",
            "fixed term", "interim",
        ],
        "entry_level": [
            "entry level", "entry-level", "fresh graduate", "fresher",
            "junior", "trainee", "intern",
        ],
        "management": [
            "head of", "director", "vp ", "vice president", "chief",
            "cto", "ceo", "cfo", "coo", "executive",
            "program manager", "general manager", "operations manager",
            "engineering manager",
        ],
        "data_science": [
            "data scientist", "data science", "machine learning", "ml engineer",
            "data analyst", "analytics", "deep learning", "ai engineer",
            "data engineer", "nlp", "computer vision",
        ],
        "software_engineering": [
            # Strong title signals — these are the canonical job title patterns
            "software engineer", "test engineer", "qa engineer",
            "developer", "backend", "frontend", "full stack", "full-stack",
            "devops", "sre", "platform engineer", "automation engineer",
            "mobile developer", "ios developer", "android developer",
            "technical analyst", "systems engineer", "cloud engineer",
            "web developer", "application developer", "site reliability",
            "test analyst", "qa lead", "test lead", "automation tester",
            "calypso developer", "calypso", "uat tester",
            "test manager", "uat manager", "uat test manager", "qa manager",
            "test architect", "principal qa",
        ],
        "finance": [
            "finance", "accounting", "banking analyst", "audit", "risk analyst",
            "compliance", "investment analyst", "actuary", "fintech",
            "quantitative analyst", "financial analyst", "treasury",
            "controller", "hedge fund",
        ],
    }

    # Senior-suppression: if title contains 'senior', drop entry_level
    is_senior = "senior" in title_lower

    # Title-weighted scoring
    scores = {}
    for category, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in title_lower:
                score += 3        # title hit — strong signal
            elif kw in body_lower:
                score += 1        # body hit — weak signal
        if is_senior and category == "entry_level":
            score = 0  # explicit guard
        scores[category] = score

    # Pick highest-scoring category
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # If we have a clear winner from the title (≥3 means at least one title hit),
    # trust the keyword pass.
    title_hit_threshold = 3
    if best_score >= title_hit_threshold:
        return best_category

    # ─── SBERT semantic fallback ────────────────────────────────
    # Only used when keyword scoring is weak (no title hit).
    # Compares JD title against canonical category descriptions.
    if sbert_model is not None and title_lower:
        try:
            import numpy as np
            CATEGORY_DESCRIPTIONS = {
                "software_engineering": "software developer engineer programmer who writes and tests code, builds applications, automates testing, develops backend or frontend systems",
                "data_science": "data scientist machine learning engineer who builds models analyzes data and uses statistics and AI",
                "management": "executive leader manager director who runs teams sets strategy and oversees operations",
                "finance": "financial analyst accountant banker who works with money trading audit risk compliance",
                "entry_level": "junior trainee intern fresh graduate with little professional experience",
                "contract": "short-term contractor freelance worker on fixed-term temporary engagement",
            }
            title_emb = sbert_model.encode(title_lower)
            best_cat = "default"
            best_sim = -1.0
            for cat, desc in CATEGORY_DESCRIPTIONS.items():
                desc_emb = sbert_model.encode(desc)
                sim = float(np.dot(title_emb, desc_emb) / (
                    np.linalg.norm(title_emb) * np.linalg.norm(desc_emb) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_cat = cat
            # Only trust SBERT if similarity is meaningful
            if best_sim >= 0.30:
                return best_cat
        except Exception:
            pass

    # If we have a body-text best with at least 2 hits, accept it
    if best_score >= 2:
        return best_category

    return "default"


def get_optimized_weights(jd_title: str, jd_text: str, sbert_model=None):
    """Get GA-optimized weights for detected job category."""
    category = detect_job_category(jd_title, jd_text, sbert_model=sbert_model)
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
