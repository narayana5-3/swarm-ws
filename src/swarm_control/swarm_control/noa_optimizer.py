"""
Generic core of the NOA + DHA + CDL-LOBL + SQP optimizer, decoupled from
path-planning specifics.

This exists so path_planning.py's planner and the CEC2017 benchmark runner
(benchmark_cec2017.py) provably run the EXACT SAME algorithm on different
objective functions -- not two separately-written implementations that
happen to share a name. That matters for the methodology: "validated on
CEC2017, then applied to path planning" only means something if it's
actually the same optimizer both times.

See path_planning.py's module docstring for the full account of which
parts of Luo et al. 2026 (INOA-SQP) this keeps vs. modifies.
"""

import numpy as np


class NOAOptimizerConfig:
    def __init__(self,
                 pop_size=40, max_iters=150,
                 ema_alpha_base=0.7, ema_c_alpha=0.4,
                 temp_cooling_rate=0.97, temp_c1=0.5, temp_c2=1.5, temp_gamma=1.5,
                 elite_ratio=0.15,
                 tent_alpha=0.7, lobl_k_max=2.5, lobl_k_min=1.0, chaos_sigma=0.3,
                 stagnation_rounds_for_lobl=5,
                 sqp_stagnation_threshold=15, sqp_elite_count=2, sqp_max_iters=15):
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.ema_alpha_base = ema_alpha_base
        self.ema_c_alpha = ema_c_alpha
        self.temp_cooling_rate = temp_cooling_rate
        self.temp_c1 = temp_c1
        self.temp_c2 = temp_c2
        self.temp_gamma = temp_gamma
        self.elite_ratio = elite_ratio
        self.tent_alpha = tent_alpha
        self.lobl_k_max = lobl_k_max
        self.lobl_k_min = lobl_k_min
        self.chaos_sigma = chaos_sigma
        self.stagnation_rounds_for_lobl = stagnation_rounds_for_lobl
        self.sqp_stagnation_threshold = sqp_stagnation_threshold
        self.sqp_elite_count = sqp_elite_count
        self.sqp_max_iters = sqp_max_iters


class NOAOptimizer:
    """
    Minimizes cost_fn(x) for x in [lo, hi]^dim.

    init_population: optional callable(rng, pop_size, dim, lo, hi) -> array
    of shape (pop_size, dim). Defaults to uniform random init. path_planning.py
    passes its own straight-line-guided initializer here; the CEC2017
    benchmark uses the default (matching the paper's own use of plain
    random initialization for black-box benchmark functions, Section 5.2).
    """

    def __init__(self, cost_fn, lo, hi, dim, config=None, rng=None, init_population=None):
        self.cost_fn = cost_fn
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        if self.lo.shape == ():
            self.lo = np.full(dim, float(self.lo))
        if self.hi.shape == ():
            self.hi = np.full(dim, float(self.hi))
        self.dim = dim
        self.cfg = config or NOAOptimizerConfig()
        self.rng = rng or np.random.default_rng(0)

        init_fn = init_population or self._default_random_init
        self.population = init_fn(self.rng, self.cfg.pop_size, self.dim, self.lo, self.hi)
        self.fitness = np.array([self.cost_fn(x) for x in self.population])

        self.best_idx = int(np.argmin(self.fitness))
        self.best = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        self.ema_memory = self.fitness.copy()
        self.local_best = self.population.copy()
        self.local_best_fitness = self.fitness.copy()
        self.stagnation_counter = np.zeros(self.cfg.pop_size, dtype=int)
        self.global_stagnation = 0
        self.tent_state = 0.35

        self.history = []  # best_fitness per iteration, for convergence plots

    def _default_random_init(self, rng, pop_size, dim, lo, hi):
        return rng.uniform(lo, hi, size=(pop_size, dim))

    def _update_ema_and_temperature(self, iteration):
        cfg = self.cfg
        diffs = np.abs(self.fitness - self.ema_memory)
        max_diff = diffs.max() + 1e-9
        alpha = cfg.ema_alpha_base - cfg.ema_c_alpha * (diffs / max_diff)
        self.ema_memory = alpha * self.ema_memory + (1 - alpha) * self.fitness

        ranks = np.argsort(np.argsort(self.ema_memory))
        rank_ratio = ranks / max(cfg.pop_size - 1, 1)

        t_base = (cfg.temp_cooling_rate ** iteration)
        temperatures = t_base * (cfg.temp_c1 + cfg.temp_c2 * (rank_ratio ** cfg.temp_gamma))
        elite_mask = rank_ratio <= cfg.elite_ratio
        return temperatures, elite_mask

    def _metropolis_accept(self, delta_e, temperature):
        if delta_e <= 0:
            return True
        return self.rng.random() < np.exp(-delta_e / max(temperature, 1e-9))

    def _tent_next(self):
        z = self.tent_state
        a = self.cfg.tent_alpha
        self.tent_state = z / a if z < a else (1 - z) / (1 - a)
        return self.tent_state

    def _lobl_opposition(self, x, iteration):
        cfg = self.cfg
        z = self._tent_next()
        k = cfg.lobl_k_max - (cfg.lobl_k_max - cfg.lobl_k_min) * (iteration / self.cfg.max_iters) \
            + cfg.chaos_sigma * (2 * z - 1)
        k = np.clip(k, cfg.lobl_k_min, cfg.lobl_k_max)
        center = (self.lo + self.hi) / 2
        opposite = center + center / k - x / k
        return np.clip(opposite, self.lo, self.hi)

    def _sqp_refine(self, x):
        from scipy.optimize import minimize
        bounds = list(zip(self.lo, self.hi))
        result = minimize(self.cost_fn, x, method="SLSQP", bounds=bounds,
                           options={"maxiter": self.cfg.sqp_max_iters, "ftol": 1e-6})
        return result.x if result.success or result.fun < self.cost_fn(x) else x

    def run(self, max_iters=None, verbose=False, verbose_every=20):
        max_iters = max_iters or self.cfg.max_iters
        cfg = self.cfg

        for iteration in range(max_iters):
            temperatures, elite_mask = self._update_ema_and_temperature(iteration)

            for i in range(cfg.pop_size):
                x = self.population[i]

                if elite_mask[i] and self.rng.random() < 0.6:
                    ref_idx = self.rng.choice(np.where(elite_mask)[0])
                    target = self.local_best[ref_idx]
                else:
                    target = self.best
                pull = self.rng.uniform(0.2, 0.8)
                noise = self.rng.normal(0, 1, size=self.dim) * (1 - iteration / max_iters) * \
                    np.linalg.norm(self.hi - self.lo) * 0.05
                candidate = x + pull * (target - x) + noise
                candidate = np.clip(candidate, self.lo, self.hi)

                cand_fitness = self.cost_fn(candidate)
                delta_e = cand_fitness - self.fitness[i]

                if self._metropolis_accept(delta_e, temperatures[i]):
                    self.population[i] = candidate
                    self.fitness[i] = cand_fitness
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                if self.fitness[i] < self.local_best_fitness[i]:
                    self.local_best[i] = self.population[i].copy()
                    self.local_best_fitness[i] = self.fitness[i]

                if self.stagnation_counter[i] >= cfg.stagnation_rounds_for_lobl:
                    opp = self._lobl_opposition(self.population[i], iteration)
                    opp_fitness = self.cost_fn(opp)
                    if opp_fitness < self.fitness[i]:
                        self.population[i] = opp
                        self.fitness[i] = opp_fitness
                    self.stagnation_counter[i] = 0

            gen_best_idx = int(np.argmin(self.fitness))
            if self.fitness[gen_best_idx] < self.best_fitness:
                self.best = self.population[gen_best_idx].copy()
                self.best_fitness = self.fitness[gen_best_idx]
                self.global_stagnation = 0
            else:
                self.global_stagnation += 1

            if self.global_stagnation >= cfg.sqp_stagnation_threshold:
                elite_indices = np.argsort(self.fitness)[:cfg.sqp_elite_count]
                for idx in elite_indices:
                    refined = self._sqp_refine(self.population[idx])
                    refined_fitness = self.cost_fn(refined)
                    if refined_fitness < self.fitness[idx]:
                        self.population[idx] = refined
                        self.fitness[idx] = refined_fitness
                        if refined_fitness < self.best_fitness:
                            self.best = refined.copy()
                            self.best_fitness = refined_fitness
                self.global_stagnation = 0

            self.history.append(float(self.best_fitness))
            if verbose and (iteration % verbose_every == 0 or iteration == max_iters - 1):
                print(f"  iter {iteration}/{max_iters}  best_cost={self.best_fitness:.6g}")

        return self.best, self.best_fitness


if __name__ == "__main__":
    # smoke test: minimize a simple known function (sphere: sum(x^2), min=0 at x=0)
    def sphere(x):
        return float(np.sum(x ** 2))

    opt = NOAOptimizer(sphere, lo=-10, hi=10, dim=5,
                        config=NOAOptimizerConfig(pop_size=20, max_iters=80))
    best_x, best_f = opt.run(verbose=True, verbose_every=20)
    print(f"\nfinal best_f={best_f:.6g} (should be close to 0)")
    assert best_f < 1.0, "optimizer failed to converge on a trivial sphere function"
    print("OK")
