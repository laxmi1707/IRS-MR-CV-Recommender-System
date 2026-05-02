"""
ga_optimizer.py — Step 5: Genetic Algorithm Weight Optimization
Offline Calibration via DEAP-style GA

Chromosome: [w_skills, w_experience, w_education, w_availability, w_misc]
Fitness:    Kendall Tau rank correlation (higher = better)

Hyperparameters (tuned via benchmark on the GroundTruth spreadsheet):
    Population:       80
    Generations:      60 (max)
    Selection:        Tournament k=3
    Crossover:        BLX-α (α=0.5), prob=0.75
    Mutation:         Gaussian σ=0.08, per-gene rate=0.15
    Elitism:          top-2 preserved each generation
    Stopping factor:  early stop when fitness plateaus for 12 consecutive gens

Pre-trained CATEGORY_WEIGHTS are static lookups — no GA execution per request.
"""

import random
import math
import numpy as np
from scipy.stats import kendalltau


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY_WEIGHTS — recalibrated per-category weight defaults.
#
# Original (sc1) weights had several anomalies:
#   - data_science:        Tech=0.2252  ← too low for an ML role
#   - software_engineering: Avail=0.30+ ← too high; corrected to 0.08
#   - finance:             Misc=0.31    ← over-weighted; rebalanced
#   - management:          Tech=0.13, Misc=0.31 ← softer signals dominated
#
# These were originally produced by GA training on a small ground-truth set;
# the optimizer found local optima that don't generalize. The values below
# are anchored to common HR-rubric expectations and verified to behave well
# in head-to-head benchmark on the same ground truth (see CHANGES doc).
#
# Each row sums to 1.0 (asserted at module load).
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    # Data Science / ML / AI — skills dominate; education matters because of
    # the math/statistics gating; availability less critical.
    "data_science": {
        "technical_skills": 0.40,
        "experience":       0.20,
        "education":        0.20,
        "availability":     0.10,
        "miscellaneous":    0.10,
    },
    # Software Engineering — Tech > Exp > Edu > Misc > Avail.
    "software_engineering": {
        "technical_skills": 0.35,
        "experience":       0.25,
        "education":        0.20,
        "miscellaneous":    0.12,
        "availability":     0.08,
    },
    # Senior IC / Lead — experience is the primary signal; skills second.
    # NEW category — sc1 had no entry, senior roles fell through to default.
    "senior": {
        "technical_skills": 0.28,
        "experience":       0.32,
        "education":        0.15,
        "miscellaneous":    0.13,
        "availability":     0.12,
    },
    # Contract / Freelance — availability is the primary signal; education
    # less relevant for short-term delivery roles.
    "contract": {
        "technical_skills": 0.32,
        "experience":       0.23,
        "education":        0.12,
        "availability":     0.22,
        "miscellaneous":    0.11,
    },
    # Finance / Banking / Compliance — experience and education both critical
    # in a regulated industry; CFA/CPA matters under education.
    "finance": {
        "technical_skills": 0.20,
        "experience":       0.30,
        "education":        0.25,
        "availability":     0.12,
        "miscellaneous":    0.13,
    },
    # Management / Leadership — experience and leadership signals (misc) lead;
    # availability less critical since exec hiring has long lead time.
    "management": {
        "technical_skills": 0.18,
        "experience":       0.28,
        "education":        0.20,
        "availability":     0.14,
        "miscellaneous":    0.20,
    },
    # Entry Level / Grad / Intern — education and skills (potential) lead;
    # experience expectation is low.
    "entry_level": {
        "technical_skills": 0.28,
        "experience":       0.12,
        "education":        0.28,
        "availability":     0.15,
        "miscellaneous":    0.17,
    },
    # Default — safe balanced weights when category is unknown.
    "default": {
        "technical_skills": 0.35,
        "experience":       0.25,
        "education":        0.15,
        "availability":     0.10,
        "miscellaneous":    0.15,
    },
}

# Sanity guard: every row must sum to 1.0 (within fp tolerance)
for _cat, _w in CATEGORY_WEIGHTS.items():
    _s = round(sum(_w.values()), 6)
    assert abs(_s - 1.0) < 1e-4, f"CATEGORY_WEIGHTS['{_cat}'] sums to {_s}, expected 1.0"

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
        # Senior IC / Lead — explicit category. Senior roles need experience-
        # weighted scoring (32%), not the default 25%. Routed via title hits.
        "senior": [
            "senior", "sr ", "lead engineer", "lead developer",
            "lead software", "principal", "staff engineer",
            "tech lead", "technical lead", "team lead",
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

    # Tie-breaker: when 'senior' is the top category but a domain category
    # (software_engineering, data_science, finance) has the same score, prefer
    # the domain category. 'Senior Software Engineer' should route to
    # 'software_engineering', not the generic 'senior' bucket — the role kind
    # carries more domain-specific weight implications than the seniority
    # modifier alone. Only fall back to 'senior' when no domain category fires.
    if best_category == "senior":
        DOMAIN_CATS = ("software_engineering", "data_science", "finance",
                       "management", "contract", "entry_level")
        for dc in DOMAIN_CATS:
            if scores.get(dc, 0) == best_score and best_score >= 3:
                best_category = dc
                break

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
                "senior": "senior staff principal lead engineer with deep experience and technical authority",
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
    n_generations: int = 60,
    population_size: int = 80,
    early_stop_patience: int = 12,
):
    """Full GA optimization. Offline phase — once per job category.

    Tuned hyper-parameter values (informed by benchmark on the GroundTruth
    spreadsheet with σ=8 noise injected into dimension scores; full details
    in CHANGES doc):

      • population_size      80   — marginally better than 50, more stable
      • n_generations        60   — gives larger population time to converge
      • early_stop_patience  12   — avoids premature stop on noisy fits
      • tournament_k          3   — low selection pressure, better exploration
                                    (k=5/7/10 essentially tied within 1 σ)
      • crossover_prob       0.75 — wide range (0.5-0.9) gave same Tau
      • mutation_rate        0.15 — best across the sweep; 0.20 nearly same
      • mutation_sigma       0.08 — finer perturbation than the 0.10 default
      • elitism             top-2 — guarantees the GA never loses its best

    Stopping factor: the GA stops when fitness plateaus for `early_stop_patience`
    generations OR when `n_generations` is reached, whichever comes first.

    Returns a dict mapping dimension name → optimized weight (sum to 1.0).
    """
    dim_keys = ["technical_skills", "experience", "education",
                "availability", "miscellaneous"]

    # ── Hyper-parameters ──────────────────────────────────────────────
    TOURNAMENT_K   = 3
    CROSSOVER_PROB = 0.75
    MUTATION_RATE  = 0.15
    MUTATION_SIGMA = 0.08
    ELITISM_N      = 2     # carry top-N forward unchanged each generation
    WEIGHT_LOW     = 0.05  # hard floor per dimension
    WEIGHT_HIGH    = 0.60  # hard ceiling per dimension

    # ── Initialise population on the simplex ──────────────────────────
    population = []
    for _ in range(population_size):
        individual = [random.uniform(WEIGHT_LOW, WEIGHT_HIGH) for _ in range(5)]
        # normalize_weights mutates in-place; capture the result by re-reading
        normalize_weights(individual)
        population.append(individual)

    best_fitness = -1.0
    best_individual = population[0][:]
    stall_count = 0

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

        # ── Stopping factor: plateau for `early_stop_patience` generations
        if stall_count >= early_stop_patience:
            break

        # ── Elitism: carry top-N unchanged ──────────────────────────────
        sorted_idx = np.argsort(fitnesses)[::-1]
        elite = [population[i][:] for i in sorted_idx[:ELITISM_N]]

        # ── Tournament selection (k=TOURNAMENT_K) ──────────────────────
        def tournament_select():
            tournament = random.sample(range(population_size), k=TOURNAMENT_K)
            winner = max(tournament, key=lambda i: fitnesses[i])
            return population[winner][:]

        # ── Build offspring with elitism + crossover + mutation ────────
        offspring = list(elite)  # seed with elites
        while len(offspring) < population_size:
            p1 = tournament_select()
            p2 = tournament_select()

            # BLX-α crossover (α=0.5)
            if random.random() < CROSSOVER_PROB:
                child1, child2 = [], []
                for j in range(5):
                    alpha = 0.5
                    d = abs(p1[j] - p2[j])
                    low = max(WEIGHT_LOW, min(p1[j], p2[j]) - alpha * d)
                    high = min(WEIGHT_HIGH, max(p1[j], p2[j]) + alpha * d)
                    child1.append(random.uniform(low, high))
                    child2.append(random.uniform(low, high))
            else:
                child1, child2 = p1[:], p2[:]

            # Gaussian mutation (σ=MUTATION_SIGMA), per-gene rate=MUTATION_RATE
            for child in (child1, child2):
                for j in range(5):
                    if random.random() < MUTATION_RATE:
                        child[j] = max(WEIGHT_LOW,
                                       min(WEIGHT_HIGH,
                                           child[j] + random.gauss(0, MUTATION_SIGMA)))

            offspring.append(child1)
            if len(offspring) < population_size:
                offspring.append(child2)

        # Normalize entire population to the simplex
        for ind in offspring:
            normalize_weights(ind)

        population = offspring[:population_size]

    normalize_weights(best_individual)
    return {dim_keys[i]: round(best_individual[i], 4) for i in range(5)}
