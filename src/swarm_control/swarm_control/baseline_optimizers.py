"""
Baseline optimizers for comparison against the NOA-derived planner --
Grey Wolf Optimizer (GWO) and Particle Swarm Optimization (PSO), the two
strongest performers against INOA-SQP in the base paper's own CEC2017
results (Table 4(a)/(b)), so comparing against them here is a fair,
literature-consistent choice rather than picking easy-to-beat baselines.

Both follow the same minimal interface as NOAOptimizer (noa_optimizer.py):
constructed with (cost_fn, lo, hi, dim), run() returns (best_x, best_f), and
.history accumulates best-fitness-per-iteration for convergence plots.
"""

import numpy as np


class GWOOptimizer:
    """Grey Wolf Optimizer (Mirjalili et al. 2014). Wolves converge on the
    3 best-known positions (alpha/beta/delta) with a linearly-decaying
    exploration coefficient."""

    def __init__(self, cost_fn, lo, hi, dim, pop_size=40, max_iters=150, rng=None):
        self.cost_fn = cost_fn
        self.lo = np.full(dim, lo) if np.isscalar(lo) else np.asarray(lo, dtype=np.float64)
        self.hi = np.full(dim, hi) if np.isscalar(hi) else np.asarray(hi, dtype=np.float64)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.rng = rng or np.random.default_rng(0)
        self.history = []

        self.population = self.rng.uniform(self.lo, self.hi, size=(pop_size, dim))
        self.fitness = np.array([cost_fn(x) for x in self.population])

    def run(self, max_iters=None, verbose=False, verbose_every=20):
        max_iters = max_iters or self.max_iters
        for t in range(max_iters):
            order = np.argsort(self.fitness)
            alpha, beta, delta = [self.population[order[i]].copy() for i in range(3)]

            a = 2 - 2 * t / max_iters  # linearly decays 2 -> 0

            for i in range(self.pop_size):
                new_pos = np.zeros(self.dim)
                for leader in (alpha, beta, delta):
                    r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - self.population[i])
                    new_pos += leader - A * D
                new_pos /= 3.0
                new_pos = np.clip(new_pos, self.lo, self.hi)

                new_fitness = self.cost_fn(new_pos)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness

            best_now = float(self.fitness.min())
            self.history.append(best_now if not self.history else min(self.history[-1], best_now))
            if verbose and (t % verbose_every == 0 or t == max_iters - 1):
                print(f"  [GWO] iter {t}/{max_iters}  best_cost={self.history[-1]:.6g}")

        best_idx = int(np.argmin(self.fitness))
        return self.population[best_idx], self.fitness[best_idx]


class PSOOptimizer:
    """Particle Swarm Optimization with inertia weight (Shi & Eberhart 1998
    style, linearly-decaying inertia)."""

    def __init__(self, cost_fn, lo, hi, dim, pop_size=40, max_iters=150, rng=None,
                 w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
        self.cost_fn = cost_fn
        self.lo = np.full(dim, lo) if np.isscalar(lo) else np.asarray(lo, dtype=np.float64)
        self.hi = np.full(dim, hi) if np.isscalar(hi) else np.asarray(hi, dtype=np.float64)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.rng = rng or np.random.default_rng(0)
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2
        self.history = []

        self.position = self.rng.uniform(self.lo, self.hi, size=(pop_size, dim))
        span = self.hi - self.lo
        self.velocity = self.rng.uniform(-span, span, size=(pop_size, dim)) * 0.1
        self.fitness = np.array([cost_fn(x) for x in self.position])

        self.pbest = self.position.copy()
        self.pbest_fitness = self.fitness.copy()
        gbest_idx = int(np.argmin(self.fitness))
        self.gbest = self.position[gbest_idx].copy()
        self.gbest_fitness = self.fitness[gbest_idx]

    def run(self, max_iters=None, verbose=False, verbose_every=20):
        max_iters = max_iters or self.max_iters
        for t in range(max_iters):
            w = self.w_max - (self.w_max - self.w_min) * t / max_iters

            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))
            self.velocity = (w * self.velocity
                              + self.c1 * r1 * (self.pbest - self.position)
                              + self.c2 * r2 * (self.gbest - self.position))
            self.position = np.clip(self.position + self.velocity, self.lo, self.hi)

            self.fitness = np.array([self.cost_fn(x) for x in self.position])
            improved = self.fitness < self.pbest_fitness
            self.pbest[improved] = self.position[improved]
            self.pbest_fitness[improved] = self.fitness[improved]

            gen_best_idx = int(np.argmin(self.fitness))
            if self.fitness[gen_best_idx] < self.gbest_fitness:
                self.gbest = self.position[gen_best_idx].copy()
                self.gbest_fitness = self.fitness[gen_best_idx]

            self.history.append(float(self.gbest_fitness))
            if verbose and (t % verbose_every == 0 or t == max_iters - 1):
                print(f"  [PSO] iter {t}/{max_iters}  best_cost={self.gbest_fitness:.6g}")

        return self.gbest, self.gbest_fitness


if __name__ == "__main__":
    # smoke test both against a trivial sphere function
    def sphere(x):
        return float(np.sum(x ** 2))

    for name, cls in [("GWO", GWOOptimizer), ("PSO", PSOOptimizer)]:
        opt = cls(sphere, lo=-10, hi=10, dim=5, pop_size=20, max_iters=80)
        best_x, best_f = opt.run(verbose=True, verbose_every=20)
        print(f"{name} final best_f={best_f:.6g} (should be close to 0)")
        assert best_f < 1.0, f"{name} failed to converge on a trivial sphere function"
    print("OK")
