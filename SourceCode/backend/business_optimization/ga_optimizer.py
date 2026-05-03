"""
ga_optimizer.py — Genetic Algorithm Weight Optimization

  • MAE-anchored stopping criterion  (new requirement)
  • A composite fitness   0.7 * (1 - MAE/100) + 0.3 * τ
    that retains rank fidelity while pulling MAE down
  • Per-generation history (best-MAE / weights / stall) for evaluation
    deliverables (Excel exports, GA convergence plots)
  • A pure-MAE mode used by ga_evaluation.MAEBasedGAOptimizer


Hyper-parameters (validated on the GroundTruth corpus — 120 rows × 8 JDs):
    Population:               80
    Generations:              200 (max)
    Selection:                Tournament k=3
    Crossover:                BLX-α (α=0.5), prob=0.75
    Mutation:                 Gaussian σ=0.08, per-gene rate=0.15
    Elitism:                  top-2 preserved each generation
    Stopping factor (MAE):    plateau ≥ 25 gens with Δ MAE < 1e-4
                              OR  best_mae ≤ target_mae   (default 0.0)
                              OR  generation count reaches max
    Stopping factor (τ):      plateau ≥ 12 gens (legacy mode only)
"""

import random
import math
import numpy as np
from scipy.stats import kendalltau


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY_WEIGHTS — production lookups (UNCHANGED from v10.4).
# These are loaded into scoring_engine.py via get_optimized_weights() at
# request time. Changing this dict changes runtime behaviour for everyone.
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

DIM_KEYS = ["technical_skills", "experience", "education",
            "availability", "miscellaneous"]
DEFAULT_WEIGHTS_LIST = [0.35, 0.25, 0.20, 0.10, 0.10]


# ═══════════════════════════════════════════════════════════════
# JD CATEGORY DETECTION 
# ═══════════════════════════════════════════════════════════════
def detect_job_category(jd_title: str, jd_text: str, sbert_model=None) -> str:
    """Infer job category from JD title and body for weight lookup.

    Two-pass approach:
      1. Keyword scoring with TITLE weighted 3× over body text.
      2. SBERT semantic fallback when keyword scoring is weak (no title hit).

    Returns: one of CATEGORY_WEIGHTS keys.
    """
    title_lower = (jd_title or "").lower().strip()
    body_lower = (jd_text or "").lower()

    CATEGORY_KEYWORDS = {
        "contract": [
            "contract", "freelance", "temporary", "6 month", "12 month",
            "fixed term", "interim",
        ],
        "entry_level": [
            "entry level", "entry-level", "fresh graduate", "fresher",
            "junior", "trainee", "intern",
        ],
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

    is_senior = "senior" in title_lower
    scores = {}
    for category, kws in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in title_lower:
                score += 3
            elif kw in body_lower:
                score += 1
        if is_senior and category == "entry_level":
            score = 0
        scores[category] = score

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # Tie-breaker: domain category beats generic 'senior' when scores tie.
    if best_category == "senior":
        DOMAIN_CATS = ("software_engineering", "data_science", "finance",
                       "management", "contract", "entry_level")
        for dc in DOMAIN_CATS:
            if scores.get(dc, 0) == best_score and best_score >= 3:
                best_category = dc
                break

    if best_score >= 3:
        return best_category

    # SBERT semantic fallback when no title hit.
    if sbert_model is not None and title_lower:
        try:
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
            if best_sim >= 0.30:
                return best_cat
        except Exception:
            pass

    if best_score >= 2:
        return best_category
    return "default"


def get_optimized_weights(jd_title: str, jd_text: str, sbert_model=None):
    """Get GA-optimized weights for the detected job category.

    Returns: (weights_dict, category_name) — the dict matches the format
    expected by scoring_engine.rank_candidates(custom_weights=...).
    """
    category = detect_job_category(jd_title, jd_text, sbert_model=sbert_model)
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["default"])
    return weights, category


# ═══════════════════════════════════════════════════════════════
# GA ENGINE 
# ═══════════════════════════════════════════════════════════════

def normalize_weights(individual):
    """Ensure weights sum to 1.0 (in-place, also returned)."""
    total = sum(individual)
    if total <= 0:
        individual[:] = DEFAULT_WEIGHTS_LIST[:]
    else:
        individual[:] = [w / total for w in individual]
    return individual


def predicted_overall(candidate_dim_scores, weights):
    """Linear weighted overall score from a 5-dim score record.

    Accepts both list-form ([s1..s5] in DIM_KEYS order) and dict-form
    ({'technical_skills': s1, ...}) so the caller can use whichever is
    convenient.
    """
    if isinstance(candidate_dim_scores, dict):
        return sum(candidate_dim_scores.get(DIM_KEYS[i], 0.0) * weights[i]
                   for i in range(5))
    return sum(candidate_dim_scores[i] * weights[i] for i in range(5))


def evaluate_fitness(
    individual,
    candidate_scores,
    ground_truth_overall=None,
    ground_truth_ranks=None,
    fitness_mode: str = "composite",
    mae_weight: float = 0.7,
    tau_weight: float = 0.3,
):
    """Evaluate a weight vector against the ground truth.

    Returns:
      (fitness, mae, tau)
        fitness — single scalar to MAXIMISE (higher is always better)
        mae     — MAE in 0–100 score units (lower is better)
        tau     — Kendall τ in [-1, 1] (higher is better)

    fitness_mode:
      "composite"  — 0.7·(1-MAE/100) + 0.3·τ        ← recommended default
      "mae"        — fitness = -MAE                 ← pure MAE minimisation
      "kendall"    — fitness = τ                    ← v10.4 legacy
    """
    individual = normalize_weights(list(individual))

    # Predicted overall per candidate
    predicted = [predicted_overall(c, individual) for c in candidate_scores]

    # MAE term
    mae = float("nan")
    if ground_truth_overall is not None and len(ground_truth_overall) == len(predicted):
        mae = float(np.mean([abs(predicted[i] - ground_truth_overall[i])
                             for i in range(len(predicted))]))

    # τ term
    tau = 0.0
    if ground_truth_ranks is not None and len(ground_truth_ranks) == len(predicted):
        try:
            pred_order = np.argsort(np.argsort([-s for s in predicted]))
            res = kendalltau(pred_order, np.array(ground_truth_ranks))
            t = float(res[0])
            if not math.isnan(t):
                tau = t
        except Exception:
            pass

    # Compose fitness per requested mode
    if fitness_mode == "mae":
        if math.isnan(mae):
            fitness = -1e9   # invalid input → very bad fitness
        else:
            fitness = -mae   # maximisation convention
    elif fitness_mode == "kendall":
        fitness = tau
    else:  # "composite"
        mae_term = max(0.0, 1.0 - (mae / 100.0)) if not math.isnan(mae) else 0.0
        fitness = mae_weight * mae_term + tau_weight * tau

    return (fitness, mae, tau)


def run_ga_optimization(
    candidate_scores,
    ground_truth_overall=None,
    ground_truth_ranks=None,
    n_generations: int = 200,
    population_size: int = 80,
    early_stop_patience: int = 25,
    target_mae: float = 0.0,
    mae_improvement_eps: float = 1e-4,
    fitness_mode: str = "composite",
    mae_weight: float = 0.7,
    tau_weight: float = 0.3,
    seed: int = 42,
    verbose: bool = False,
):
    """Full GA optimisation — MAE-anchored stopping.

    Args:
      candidate_scores:        list[dict|list]  each candidate's 5 dim scores
                               keyed by DIM_KEYS or in DIM_KEYS order
      ground_truth_overall:    list[float]      HR-assigned overall (0–100) per
                                                candidate (required for MAE)
      ground_truth_ranks:      list[int]        optional HR rank (1=best) per
                                                candidate (required for τ)
      n_generations:           int              hard maximum
      population_size:         int              GA population
      early_stop_patience:     int              halt after this many gens
                                                of no MAE improvement
      target_mae:              float            halt when best MAE ≤ this
      mae_improvement_eps:     float            improvements smaller than
                                                this don't count
      fitness_mode:            "composite"|"mae"|"kendall"
      mae_weight, tau_weight:  composite blend  (only used in composite mode)

    Returns:
      {
        "weights":             dict mapping DIM_KEY → weight (sums to 1.0),
        "best_mae":            float,
        "best_tau":            float,
        "best_fitness":        float,
        "history":             list of per-generation snapshots,
        "stopped_because":     "plateau" | "target_mae_reached" | "max_generations",
        "generations_run":     int,
      }

    Backward compatibility:
      The v10.4 signature was run_ga_optimization(candidate_scores,
      ground_truth_ranks, n_generations, population_size, early_stop_patience).
      Callers using positional ranks still work — see the second-positional
      conversion below.
    """
    if (ground_truth_overall is not None and ground_truth_ranks is None
            and all(isinstance(x, int) and x > 0 for x in ground_truth_overall)):
        ground_truth_ranks = ground_truth_overall
        ground_truth_overall = None
        if fitness_mode == "composite":
            # No MAE signal available → fall back to pure τ
            fitness_mode = "kendall"

    random.seed(seed)
    np.random.seed(seed)

    # ── Hyperparameters (v10.4 sweet spot, retained) ──────────────────
    TOURNAMENT_K   = 3
    CROSSOVER_PROB = 0.75
    MUTATION_RATE  = 0.15
    MUTATION_SIGMA = 0.08
    ELITISM_N      = 2
    WEIGHT_LOW     = 0.05
    WEIGHT_HIGH    = 0.60

    # ── Initialise population on the simplex ──────────────────────────
    population = []
    for _ in range(population_size):
        individual = [random.uniform(WEIGHT_LOW, WEIGHT_HIGH) for _ in range(5)]
        normalize_weights(individual)
        population.append(individual)

    best_fitness = -float("inf")
    best_mae = float("inf")
    best_tau = -1.0
    best_individual = population[0][:]
    stall_count = 0
    history = []
    stopped_because = "max_generations"

    for gen in range(n_generations):
        evals = [
            evaluate_fitness(ind, candidate_scores,
                             ground_truth_overall, ground_truth_ranks,
                             fitness_mode, mae_weight, tau_weight)
            for ind in population
        ]
        fitnesses = [e[0] for e in evals]
        maes      = [e[1] for e in evals]
        taus      = [e[2] for e in evals]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = float(fitnesses[gen_best_idx])
        gen_best_mae = float(maes[gen_best_idx])
        gen_best_tau = float(taus[gen_best_idx])

        # Headline objective is MAE — track best by lowest MAE, with fitness
        # used as tiebreaker when MAE is unavailable.
        improved = False
        if not math.isnan(gen_best_mae):
            if gen_best_mae < best_mae - mae_improvement_eps:
                improved = True
        else:
            if gen_best_fitness > best_fitness + 1e-5:
                improved = True

        if improved:
            best_fitness = gen_best_fitness
            best_mae = gen_best_mae
            best_tau = gen_best_tau
            best_individual = population[gen_best_idx][:]
            stall_count = 0
        else:
            stall_count += 1

        history.append({
            "generation": gen + 1,
            "best_mae": round(best_mae, 4) if not math.isnan(best_mae) else None,
            "gen_best_mae": round(gen_best_mae, 4) if not math.isnan(gen_best_mae) else None,
            "best_tau": round(best_tau, 4),
            "gen_best_tau": round(gen_best_tau, 4),
            "best_fitness": round(best_fitness, 4),
            "weights": [round(w, 4) for w in best_individual],
            "stall": stall_count,
        })

        if verbose:
            mae_disp = f"{best_mae:.4f}" if not math.isnan(best_mae) else "  N/A "
            print(f"  gen {gen+1:3d}  best_mae={mae_disp}  τ={best_tau:.3f}  "
                  f"fitness={best_fitness:.4f}  stall={stall_count}")

        # ── Stopping criteria ────────────────────────────────────────
        if not math.isnan(best_mae) and best_mae <= target_mae:
            stopped_because = "target_mae_reached"
            break
        if stall_count >= early_stop_patience:
            stopped_because = "plateau"
            break

        # ── Elitism + tournament + BLX-α + Gaussian mutation ─────────
        sorted_idx = np.argsort(fitnesses)[::-1]
        elite = [population[i][:] for i in sorted_idx[:ELITISM_N]]

        def tournament_select():
            picks = random.sample(range(population_size), k=TOURNAMENT_K)
            winner = max(picks, key=lambda i: fitnesses[i])
            return population[winner][:]

        offspring = list(elite)
        while len(offspring) < population_size:
            p1 = tournament_select()
            p2 = tournament_select()

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

            for child in (child1, child2):
                for j in range(5):
                    if random.random() < MUTATION_RATE:
                        child[j] = max(WEIGHT_LOW,
                                       min(WEIGHT_HIGH,
                                           child[j] + random.gauss(0, MUTATION_SIGMA)))

            offspring.append(child1)
            if len(offspring) < population_size:
                offspring.append(child2)

        for ind in offspring:
            normalize_weights(ind)
        population = offspring[:population_size]

    normalize_weights(best_individual)
    return {
        "weights": {DIM_KEYS[i]: round(best_individual[i], 4) for i in range(5)},
        "best_mae": round(best_mae, 4) if not math.isnan(best_mae) else None,
        "best_tau": round(best_tau, 4),
        "best_fitness": round(best_fitness, 4),
        "history": history,
        "stopped_because": stopped_because,
        "generations_run": len(history),
    }


# ═══════════════════════════════════════════════════════════════
# CACHING ADAPTER 
# ═══════════════════════════════════════════════════════════════

def run_ga_with_cached_scores(
    cached_pairs,
    fitness_mode: str = "mae",
    **ga_kwargs,
):
    """Run the GA against a pre-scored cache of (jd, candidate, dims, gt) tuples.

    This is the recommended driver for production-style optimisation runs —
    score every (JD, candidate) pair ONCE through the full ICRS pipeline,
    cache the 5-dim score vectors, then let the GA explore the weight space
    against the cached scores. Each fitness evaluation is now a sub-millisecond
    arithmetic operation, so a 200-generation run completes in seconds rather
    than hours.

    Args:
      cached_pairs:  list of dicts, each with keys:
                       "dims"   → dict of 5 dim scores  (in DIM_KEYS)
                       "hr_overall" → float (HR ground-truth overall, 0-100)
                       "hr_rank"    → int (optional, for τ)
                       "jd_id"      → str (optional metadata)
                       "candidate_id" → str (optional metadata)
      fitness_mode:  "mae" | "composite" | "kendall"
      **ga_kwargs:   any additional arguments passed through to
                     run_ga_optimization (n_generations, population_size,
                     early_stop_patience, target_mae, …)

    Returns: same shape as run_ga_optimization().
    """
    candidate_scores = [p["dims"] for p in cached_pairs]
    gt_overall = [p["hr_overall"] for p in cached_pairs]
    gt_ranks = [p.get("hr_rank") for p in cached_pairs]
    if any(r is None for r in gt_ranks):
        gt_ranks = None

    return run_ga_optimization(
        candidate_scores,
        ground_truth_overall=gt_overall,
        ground_truth_ranks=gt_ranks,
        fitness_mode=fitness_mode,
        **ga_kwargs,
    )