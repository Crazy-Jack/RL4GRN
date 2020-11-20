from random import sample
from typing import List

import numpy as np

NUM_GENES: int = 50
ACTION_SIZE: int = 10  # number of genes chosen to modify per action
EPSILON: float = 0.001
I: np.array = np.identity(NUM_GENES)
CUTOFF = 500

ODE_COEFFICIENTS: np.array = np.random.rand(NUM_GENES, NUM_GENES)  # a


def make_action() -> List[int]:
    return sample(range(NUM_GENES), ACTION_SIZE)


def backwards_euler(x_n: np.array, perturbation, delta=0.01):
    # in the event of oscillation, a cutoff will implicitly choose a state
    x_n_1 = None
    for step in range(CUTOFF):
        print(f"iter {step}: {x_n_1}")
        coeff = ODE_COEFFICIENTS * delta - I
        const = -x_n - delta * perturbation
        x_n_1 = np.linalg.solve(coeff, const)

        # stop if converged
        if np.abs(x_n_1 - x_n).sum() < EPSILON:
            print("## CONVERGED ##")
            return x_n_1
        else:
            print("total diff:", np.abs(x_n_1 - x_n).sum())
            x_n = x_n_1

    return x_n_1


if __name__ == "__main__":
    init = np.random.rand(NUM_GENES, 1)  # x
    adjust = 0 * np.random.rand(NUM_GENES, 1)  # b
    nextState = backwards_euler(init, perturbation=adjust)
