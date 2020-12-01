import logging
import time
from itertools import count

import numpy as np


class Integrator:
    def __init__(self, ode_coefficients: np.array, seed=1996):
        self.seed = seed
        self.ode_coefficients = ode_coefficients
        self.n = ode_coefficients.shape[0]
        self._validate()
        np.random.seed(seed)

    def get_next(self, x_n: np.array, perturbation: np.array):
        raise NotImplemented()

    def _validate(self):
        assert isinstance(self.ode_coefficients, np.ndarray)  # np
        assert np.issubdtype(self.ode_coefficients.dtype, np.float64)  # floats
        assert len(self.ode_coefficients) == len(self.ode_coefficients[0])  # square
        assert isinstance(self.seed, int)


class BackwardsEuler(Integrator):

    def __init__(self, ode_coefficients, delta: float = 1E-3, epsilon: float = 1E-4, cutoff: int = 15E3):
        super().__init__(ode_coefficients)
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.delta = delta
        self._steps = 0

    def get_next(self, x_n: np.array, perturbation: np.array) -> np.array:
        # in the event of oscillation, a cutoff will implicitly choose a state
        x_n_1, step = None, None
        for step in count(1):
            coeff = self.ode_coefficients * self.delta - np.identity(self.n)
            const = -x_n - self.delta * perturbation
            x_n_1 = np.linalg.solve(coeff, const)

            # stop if converged
            if np.abs(x_n_1 - x_n).sum() < self.epsilon:
                # CONVERGED
                break
            elif step == self.cutoff:
                logging.warning(f"Backwards Euler did not converge after {self.cutoff} iterations. "
                                f"Returning anyway.")
                break
            else:
                x_n = x_n_1

        self._steps = step
        return x_n_1

    def _performance_test(self):
        x_n = np.random.rand(self.n, 1)
        perturb = np.random.rand(self.n, 1)

        start = time.time()
        self.get_next(x_n, perturb)
        end = time.time()
        return end - start, self._steps

    def run_performance_testing(self, iterations=10):
        times, steps = zip(*[self._performance_test() for _ in range(iterations)])
        print("average time:", sum(times) / len(times))
        print("average steps:", sum(steps) / len(steps))


def make_symmetric_random(n):
    tmp = np.random.rand(n, n)
    sym = (tmp + tmp.T) / 2
    sym[np.triu_indices_from(sym)] = -sym[np.triu_indices_from(sym)]
    return sym


if __name__ == "__main__":
    n = 50
    initial = np.random.rand(n, 1)  # x
    perturbation = np.random.rand(n, 1)  # b
    ode_coefficients = make_symmetric_random(n)

    be = BackwardsEuler(ode_coefficients)

    print("x_{n} =", initial.round(3).ravel())
    print("perturb =", perturbation.round(3).ravel())
    print("coefficients =\n", ode_coefficients)
    print("x_{converged} =", be.get_next(initial, perturbation).round(3).ravel())
