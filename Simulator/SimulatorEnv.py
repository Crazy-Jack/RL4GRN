import numpy as np
import time

from integrators import BackwardsEuler, make_symmetric_random

class SimulatorEnv:
    """class for simulation environment
    Usage:
        env = SimulatorEnv(coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target, lambda_distance_reward)
        next_state, reward, done, info, distance_to_target = env(action)
    See `test_simulator` for concrete example.
    """

    def __init__(self, coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target,
                 lambda_distance_reward):
        self.num_genes = coefficient.shape[0]
        self.coefficient = coefficient  # np.array [num_genes, num_genes]
        self.target_state = target_state  # np.array [num_genes,]
        self.init_state = init_state  # np.array [num_genes,]
        self.time_limit = time_limit  # time limit for an episode, one episode corresponds to certain number of actions
        self.euler_limit = euler_limit  # the maximum delta_t can take for integration when calling backward euler
        self.delta = delta  # delta in integration method
        self.eps_euler = eps_euler  # epsilon used in backward Euler
        self.eps_target = eps_target  # epsilon used for deciding if  current state is close enough to  final state
        self.lambda_distance_reward = lambda_distance_reward  # lambda that controls the weight for the reward

        self.integrator = BackwardsEuler(ode_coefficients=self.coefficient,
                                         cutoff=self.euler_limit,
                                         delta=self.delta,
                                         epsilon=self.eps_euler)
        # initial state
        self.state = init_state
        self.accumulate_step = 0
        self.reset()

    def get_reward(self, next_state):
        """given next state, calculating the reward"""

        # distance to target state
        distance_to_target = ((self.target_state - next_state) ** 2).sum()

        # evaluate if the episode ends
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
        next_state = self.integrator.get_next(self.state, action)
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
    num_genes = 100
    time_limit = 200
    euler_limit = 16000
    delta = 1e-2

    eps_euler = 1e-4
    eps_target = 1e-2
    lambda_distance_reward = 0.1  # reward = - distance_to_target * lambda_distance_reward - 1

    coefficient = make_symmetric_random(num_genes)
    init_state = np.random.rand(num_genes, 1)
    target_state = np.random.rand(num_genes, 1)

    # define env
    env = SimulatorEnv(coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target,
                       lambda_distance_reward)

    for i in range(time_limit):
        # generate random action, replace this with policy network in reinforcement learning
        action = np.random.rand(num_genes, 1)
        # put action into environment for evaluation
        start_time = time.time()
        next_state, reward, done, info, distance_to_target = env.step(
            action)  # env will update its current state after taking every step
        end_time = time.time()
        time_delta_step = end_time - start_time
        print("time: {}; reward: {}; done {}; distance_to_target {}".format(time_delta_step, reward, done,
                                                                            distance_to_target))
        print(info)


if __name__ == "__main__":
    # test_backward()
    test_simulator()
