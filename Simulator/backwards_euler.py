import logging
from itertools import count

import numpy as np


def make_symmetric_random():
    tmp = np.random.rand(NUM_GENES, NUM_GENES)
    sym = (tmp + tmp.T) / 2
    sym[np.triu_indices_from(sym)] = -sym[np.triu_indices_from(sym)]
    return sym


def backwards_euler(x_n: np.array, perturbation):
    # in the event of oscillation, a cutoff will implicitly choose a state
    x_n_1 = None
    for step in count(1):
        coeff = ODE_COEFFICIENTS * DELTA - I
        const = -x_n - DELTA * perturbation
        x_n_1 = np.linalg.solve(coeff, const)

        # stop if converged
        if np.abs(x_n_1 - x_n).sum() < EPSILON:
            # CONVERGED
            break
        elif step == CUTOFF:
            logging.warning(f"Backwards Euler did not converge after {CUTOFF} iterations. "
                            f"Returning anyway.")
            break
        else:
            x_n = x_n_1

    return x_n_1


NUM_GENES: int = 50
EPSILON: float = 1E-4
DELTA = 1E-3
I: np.array = np.identity(NUM_GENES)
ODE_COEFFICIENTS: np.array = make_symmetric_random()

# for EPSILON=1E-4, DELTA=1E-3, converged on average around 12.3 (sample size: 50)
CUTOFF = 16000

if __name__ == "__main__":
    init = np.random.rand(NUM_GENES, 1)  # x
    adjust = np.random.rand(NUM_GENES, 1)  # b
    print(backwards_euler(init, adjust))
