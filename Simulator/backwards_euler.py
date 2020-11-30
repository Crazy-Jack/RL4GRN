import logging
import time
from itertools import count

import numpy as np


# np.random.seed(1996)

def make_symmetric_random(NUM_GENES):
    tmp = np.random.rand(NUM_GENES, NUM_GENES)
    sym = (tmp + tmp.T) / 2
    sym[np.triu_indices_from(sym)] = -sym[np.triu_indices_from(sym)]
    return sym


def backwards_euler(x_n, perturbation, CUTOFF, ODE_COEFFICIENTS, DELTA, EPSILON, get_steps=False):
    # in the event of oscillation, a cutoff will implicitly choose a state
    x_n_1 = None
    NUM_GENES = x_n.shape[0]
    for step in count(1):
        coeff = ODE_COEFFICIENTS * DELTA - np.identity(NUM_GENES)
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
    print(f"Converged in {step} steps")
    if get_steps:
        return step

    return x_n_1


def performance():
    # global ODE_COEFFICIENTS
    ODE_COEFFICIENTS = make_symmetric_random()
    start = time.time()
    steps = backwards_euler(np.random.rand(NUM_GENES, 1), np.random.rand(NUM_GENES, 1), get_steps=True)
    end = time.time()
    return end - start, steps




def test_backward():
    NUM_GENES: int = 50
    EPSILON: float = 1E-4
    DELTA = 1E-3
    I: np.array = np.identity(NUM_GENES)
    ODE_COEFFICIENTS: np.array = make_symmetric_random()

    # for EPSILON=1E-4, DELTA=1E-3, converged on average around 12.3 (sample size: 50)
    CUTOFF = 16000

    init = np.random.rand(NUM_GENES, 1)  # x
    adjust = np.random.rand(NUM_GENES, 1)  # b
    print("x_{n} =", init.round(3).ravel())
    print("perturb =", adjust.round(3).ravel())
    print("coefficients =\n", ODE_COEFFICIENTS)
    print("x_{converged} =", backwards_euler(init, adjust).round(3).ravel())

    times, steps = zip(*[performance() for _ in range(50)])
    print("average time:", sum(times) / len(times))
    print("average steps:", sum(steps) / len(steps))




# ==================================

class SimulatorEnv:
    """class for simulation environment
    Usage: 
        env = SimulatorEnv(coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target, lambda_distance_reward)
        next_state, reward, done, info, distance_to_target = env(action)
    See `test_simulator` for concrete example.
    """
    def __init__(self, coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target, lambda_distance_reward):
        self.num_genes = coefficient.shape[0]
        self.coefficient = coefficient # np.array [num_genes, num_genes]
        self.target_state = target_state # np.arrya [num_genes,]
        self.init_state = init_state # np.arrya [num_genes,]
        self.time_limit = time_limit # time limit for an episode, one eposode corresponds to certain number of actions
        self.euler_limit = euler_limit # the maximum delta_t can take for integration when calling backward euler
        self.delta = delta # delta in integration method 
        self.eps_euler = eps_euler # epsilon used in backward eular
        self.eps_target = eps_target # epsilon used for deciding if the current state is close enough to the final state
        self.lambda_distance_reward = lambda_distance_reward # lambda that controls the weight for the reward 

        # initial state
        self.reset()
    
    def get_reward(self, next_state):
        """given next state, calculating the reward"""

        # distance to target state
        distance_to_target = ((self.target_state - next_state) ** 2).sum()

        # eval if the episode ends
        if distance_to_target < self.eps_target:
            done = True
            info = "reach the goal (error {})".format(distance_to_target)
        elif self.accumulate_step >= self.time_limit:
            done = True
            info = "reach the end of episode {}".format(self.accumulate_step)
        else:
            done = False
            info = "keep trying, step {} / {}".format(self.accumulate_step, self.time_limit)
        
        # calculate reward
        reward = - distance_to_target * self.lambda_distance_reward - 1 
        return reward, done, info, distance_to_target

    def step(self, action):
        """take an action (perturb), output (next_state, reward, done, info)"""
        # use backward euler for integrate
        next_state = backwards_euler(self.state, action, self.euler_limit, self.coefficient, self.delta, self.eps_euler)
        # calculate next state
        reward, done, info, distance_to_target = self.get_reward(next_state)
        # update the state
        self.state = next_state
        self.accumulate_step += 1
        return next_state, reward, done, info, distance_to_target
    
    def reset(self):
        """reset the env"""
        self.state = self.init_state
        self.accumulate_step = 0
        return self.state


def test_simulator():
    # define parameters
    NUM_GENES = 100
    time_limit = 200
    euler_limit = 16000
    delta = 1e-2

    eps_euler = 1e-4
    eps_target = 1e-2
    lambda_distance_reward = 0.1 # reward = - distance_to_target * lambda_distance_reward - 1

    coefficient = make_symmetric_random(NUM_GENES)
    init_state = np.random.rand(NUM_GENES, 1)
    target_state = np.random.rand(NUM_GENES, 1)
    
    # define env
    env = SimulatorEnv(coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target, lambda_distance_reward)

    for i in range(time_limit):
        # generate random action, replace this with policy network in reinforcement learning
        action = np.random.rand(NUM_GENES, 1)
        # put action into environment for evulation
        start_time = time.time()
        next_state, reward, done, info, distance_to_target = env.step(action) # env will update its current state after taking every step
        end_time = time.time()
        time_delta_step = end_time - start_time
        print("time: {}; reward: {}; done {}; distance_to_target {}".format(time_delta_step, reward, done, distance_to_target))
        print(info)


if __name__ == "__main__":
    # test_backward()
    test_simulator()