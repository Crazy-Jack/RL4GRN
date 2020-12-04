import time

import numpy as np

from .integrators import BackwardsEuler, make_symmetric_random


# ==================================

class SimulatorEnv:
    """class for simulation environment
    Usage: 
        env = SimulatorEnv(coefficient, init_state, target_state, time_limit, euler_limit, delta, eps_euler, eps_target, lambda_distance_reward)
        next_state, reward, done, info, distance_to_target = env(action)
    See `test_simulator` for concrete example.
    """

    def __init__(self, coefficient, init_state, target_state, original_perturb,
                 action_index, max_action, time_limit, euler_limit, delta, eps_euler, eps_target,
                 lambda_distance_reward):
        self.num_genes = coefficient.shape[0]
        self.coefficient = coefficient  # np.array [num_genes, num_genes]
        self.original_perturb = original_perturb  # original perturb, if certain genes is not actionable through action, it remains the original perturbation term
        self.target_state = target_state  # np.array [num_genes,]
        self.init_state = init_state  # np.array [num_genes,]
        self.state_space = len(self.init_state)
        self.time_limit = time_limit  # time limit for an episode, one eposode corresponds to certain number of actions
        self.euler_limit = euler_limit  # the maximum delta_t can take for integration when calling backward euler
        self.delta = delta  # delta in integration method
        self.eps_euler = eps_euler  # epsilon used in backward euler
        self.eps_target = eps_target  # epsilon used for deciding if current state is close enough to final state
        self.lambda_distance_reward = lambda_distance_reward  # lambda that controls the weight for the reward

        self.integrator = BackwardsEuler(ode_coefficients=self.coefficient,
                                         cutoff=self.euler_limit,
                                         delta=self.delta,
                                         epsilon=self.eps_euler)

        # initialize state
        self.state = init_state
        self.accumulate_step = 0
        self.reset()  # reassign the state into self.init_state

        # define action space
        self.action_index = action_index  # a list, define action space index (e.g [1,3,4,8]), indicating which genes in [num_genes,] vector is actionable
        self.action_space = len(self.action_index)
        self.max_action = max_action  # scalar,

        # final check
        self.self_assert()

    def self_assert(self):
        """check if the init is reasonable"""
        assert self.action_space < self.state_space  # only selected not all of the genes can be modified
        assert self.state_space == len(self.target_state)  # target space is matched with the init state space

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
            info = "reach the end of episode. ".format(self.accumulate_step)
        else:
            done = False
            info = "keep trying, error {}, step {} / {}".format(distance_to_target, self.accumulate_step, self.time_limit)
        
        # calculate reward
        reward = - distance_to_target * self.lambda_distance_reward - 1
        return reward, done, info, distance_to_target

    def step(self, action):
        """take an action (perturb), output (next_state, reward, done, info)
        param: 
            - action: # [self.action_space, ] should be a numpy vector
        """
        # construct the resulted perturb
        assert len(action) == self.action_space
        perturb = self.original_perturb
        # print("perturb", perturb[self.action_index].shape)
        perturb[self.action_index] = action
        # use backward euler for integrate
        next_state = self.integrator.get_next(self.state, perturb)
        # calculate next state
        reward, done, info, distance_to_target = self.get_reward(next_state)
        # update the state
        self.state = next_state
        self.accumulate_step += 1
        return next_state, reward, done, info

    def reset(self):
        """reset the env"""
        self.state = self.init_state
        self.accumulate_step = 0
        return self.state

    def sample_action(self):
        """sample a random action from"""
        action = np.random.uniform(-self.max_action, self.max_action, self.action_space)
        return action
    

# store the parameters for specific environment
def make(env_name, seed_network, seed_init):
    """Automatically make environment"""

    if env_name == 'random_generate':
        # tunable parameter
        num_genes = 100
        time_limit = 200
        euler_limit = 16000
        action_percent = 0.3
        delta = 1e-2
        eps_euler = 1e-4
        eps_target = 1e-2
        lambda_distance_reward = 0.1  # reward = - distance_to_target * lambda_distance_reward - 1
        max_action = 10
        # init the network structure and the actionable genes
        np.random.seed(seed_network)
        coefficient = make_symmetric_random(num_genes)  # random
        action_space = int(num_genes * action_percent)
        action_index = np.random.randint(num_genes, size=(action_space,))  # random

        # init the initial state, the target state
        np.random.seed(seed_init)
        init_state = np.random.rand(num_genes,)  # random
        target_state = np.random.rand(num_genes, )  # random
        original_perturb = np.random.rand(num_genes, )  # random

        print(action_index.shape)
        # define env
        env = SimulatorEnv(coefficient, init_state, target_state,
                           original_perturb, action_index, max_action, time_limit, euler_limit,
                           delta, eps_euler, eps_target, lambda_distance_reward)
        return env
    elif env_name == 'random_generate_td3_simple':
        # tunable parameter
        num_genes = 10
        time_limit = 100
        euler_limit = 16000
        action_percent = 0.5
        delta = 1e-1
        eps_euler = 1e-4
        eps_target = 1e-2
        lambda_distance_reward = 0.5  # reward = - distance_to_target * lambda_distance_reward - 1
        max_action = 10
        # init the network structure and the actionable genes
        np.random.seed(seed_network)
        coefficient = make_symmetric_random(num_genes)  # random
        action_space = int(num_genes * action_percent)
        action_index = np.random.randint(num_genes, size=(action_space,))  # random

        # init the initial state, the target state
        np.random.seed(seed_init)
        init_state = np.random.rand(num_genes,)  # random
        target_state = np.random.rand(num_genes, )  # random
        original_perturb = np.random.rand(num_genes, )  # random

        print(action_index.shape)
        # define env
        env = SimulatorEnv(coefficient, init_state, target_state,
                           original_perturb, action_index, max_action, time_limit, euler_limit,
                           delta, eps_euler, eps_target, lambda_distance_reward)
        return env

    elif env_name == 'infer_from_data':
        pass

    else:
        raise NotImplementedError("Env {} not implemented. ".format(env_name))


def test_simulator():
    # define parameters
    env = make('random_generate', seed_network=0, seed_init=1)
    train_steps = 300
    episode = 0
    for i in range(train_steps):
        # generate random action, replace this with policy network in reinforcement learning
        action = np.random.rand(env.action_space, 1) * env.max_action
        # put action into environment for evaluation
        start_time = time.time()
        next_state, reward, done, info = env.step(action)  # env will update its current state after taking every step
        distance_to_target = ((next_state - env.target_state) ** 2).sum()
        end_time = time.time()
        time_delta_step = end_time - start_time
        print("time: {:.4f} s; reward: {}; done {}; distance_to_target {}; Info: {}".format(time_delta_step, reward, done,
                                                                            distance_to_target,info))
        # print(info)

        if done:
            print("Episode {} End. ----------\n".format(episode))
            env.reset()
            episode += 1


if __name__ == '__main__':
    test_simulator()
