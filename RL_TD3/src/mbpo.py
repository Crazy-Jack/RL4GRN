'''
    Main class for MBPO/TD3. Contains the training routine for both MBPO and TD3,
    as well as model rollout, evaluation, and graphing functions.
    You will implement part of this file.
'''
# pylint: disable=W0201, C0103,
import os
import numpy as np
import tensorflow as tf
import pybullet_envs
import gym
import matplotlib.pyplot as plt

from src.utils import ReplayBuffer
from src.td3 import TD3
from src.pe_model import PE
from src.fake_env import FakeEnv

import Simulator.SimulatorEnv as sim


class MBPO:
    '''
        The main class for both TD3 and MBPO. Some of the attributes are only
        used for MBPO and not for TD3. But notice that the vast majority
        of code is shared.
    '''
    def __init__(self, train_kwargs, model_kwargs, TD3_kwargs):
        # shared training parameters
        self.enable_MBPO = train_kwargs["enable_MBPO"]
        self.policy_name = train_kwargs["policy"]
        self.env_name = train_kwargs["env_name"]
        self.experiment_name = train_kwargs["experiment_name"]
        self.seed = train_kwargs["seed"] #random-seed
        self.load_model = train_kwargs["load_model"]
        self.max_timesteps = train_kwargs["max_timesteps"] #maximum real-env timestemps
        self.start_timesteps = train_kwargs["start_timesteps"] #burn-in period
        self.batch_size = train_kwargs["batch_size"]
        self.eval_freq = train_kwargs["eval_freq"] #Model evaluation frequency
        self.save_model = train_kwargs["save_model"]
        self.expl_noise = train_kwargs["expl_noise"] #TD3 exploration noise
        self.seed_coeff, self.seed_init = train_kwargs["seed_coeff"], train_kwargs["seed_init"]
        self.HER = train_kwargs["HER"] # whether to perform Hindsight Experience Replay (HER)
        self.HER_k = train_kwargs["HER_k"] # number of future goals to sample
        # MBPO parameters. Pseudocode refers to MBPO pseudocode in writeup.
        self.model_rollout_batch_size = train_kwargs["model_rollout_batch_size"]
        self.num_rollouts_per_step = train_kwargs["num_rollouts_per_step"] #M in pseudocode
        self.rollout_horizon = train_kwargs["rollout_horizon"] #k in pseudocode
        self.model_update_freq = train_kwargs["model_update_freq"] #E in pseudocode
        self.num_gradient_updates = train_kwargs["num_gradient_updates"] #G in pseudocode
        self.percentage_real_transition = train_kwargs["percentage_real_transition"]
        self.random_init_everytime = train_kwargs["random_init_everytime"] 
        # TD3 agent parameters
        self.discount = TD3_kwargs["discount"] #discount factor
        self.tau = TD3_kwargs["tau"] #target network update rate
        self.policy_noise = TD3_kwargs["policy_noise"] #sigma in Target Policy Smoothing
        self.noise_clip = TD3_kwargs["noise_clip"] #c in Target Policy Smoothing
        self.policy_freq = TD3_kwargs["policy_freq"] #d in TD3 pseudocode

        # Dynamics model parameters
        self.num_networks = model_kwargs["num_networks"] #number of networks in ensemble
        self.num_elites = model_kwargs["num_elites"] #number of elites used to predict
        self.model_lr = model_kwargs["model_lr"] #learning rate for dynamics model

        # Since dynamics model remains unchanged every epoch
        # We can perform the following optimization:
        # instead of sampling M rollouts every step for E steps, sample B * M rollouts per
        # epoch, where each epoch is just E environment steps.
        self.rollout_batch_size = self.model_rollout_batch_size * self.num_rollouts_per_step
        # Number of steps in FakeEnv
        self.fake_env_steps = 0

    def eval_policy(self, eval_episodes=50):
        '''
            Runs policy for eval_episodes and returns average reward.
            A fixed seed is used for the eval environment.
            Do not modify.
        '''
        env_name = self.env_name
        seed = self.seed
        policy = self.policy
        seed_coeff, seed_init = self.seed_coeff, self.seed_init

        eval_env = sim.make(env_name, seed_coeff, seed_init, goal_condition=self.HER, random_init_target=self.random_init_everytime)
        # eval_env = sim.make(env_name, seed_coeff, seed_init, goal_condition=self.HER)
        orginal_gap = 0.

        avg_reward = 0.
        avg_distance = 0.
        steps = 0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            gap = np.abs(state[:eval_env.state_space] - state[eval_env.state_space:]).sum()
            orginal_gap += gap
            print("init gap", gap)
            while not done:
                action = policy.select_action(np.array(state))
                # print("action shape", action.shape)
                state, reward, done, info = eval_env.step(action)
                
                print("Eval info: {}; Action max {}; min {}; mean {}".format(info, action.max(), action.min(), action.mean()))
                avg_reward += reward
                
                steps += 1

            if self.HER:
                distance_to_target = np.abs(state[:eval_env.state_space] - eval_env.target_state).sum()
            else:
                distance_to_target = np.abs(state - eval_env.target_state).sum()
            
            avg_distance += distance_to_target
            print("===============================================")
            print("===============================================")
        avg_reward /= eval_episodes
        avg_distance /= eval_episodes
        orginal_gap /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: Reward {avg_reward:.3f}; Error {avg_distance:.4f}")
        print("---------------------------------------")
        return avg_reward, avg_distance, orginal_gap

    def init_models_and_buffer(self):
        '''
            Initialize the PE dynamics model, the TD3 policy, and the two replay buffers.
            The PE dynamics model and the replay_buffer_Model will not be used if MBPO is disabled.
            Do not modify.
        '''
        self.file_name = f"{self.policy_name}_{self.env_name}_{self.experiment_name}_{self.seed}"
        print("---------------------------------------")
        print(f"Policy: {self.policy_name}, Env: {self.env_name}, Seed: {self.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results/{}".format(self.experiment_name)):
            os.makedirs("./results/{}".format(self.experiment_name))

        if self.save_model and not os.path.exists("./models/{}".format(self.experiment_name)):
            os.makedirs("./models/{}".format(self.experiment_name))

        tf.random.set_seed(self.seed)
        seed_coeff, seed_init = self.seed_coeff, self.seed_init

        env = sim.make(self.env_name, seed_coeff, seed_init)
        state_dim = env.state_space 
        if self.HER: 
            state_dim += env.goal_space
        action_dim = env.action_space
        max_action = env.max_action

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.max_action = max_action

        td3_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.discount,
            "tau": self.tau,
        }

        # Target policy smoothing is scaled wrt the action scale
        td3_kwargs["policy_noise"] = self.policy_noise * max_action
        td3_kwargs["noise_clip"] = self.noise_clip * max_action
        td3_kwargs["policy_freq"] = self.policy_freq

        model_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_networks": self.num_networks,
            "num_elites": self.num_elites,
            "learning_rate": self.model_lr,
        }

        self.policy = TD3(**td3_kwargs) #TD3 policy
        self.model = PE(**model_kwargs) #Dynamics model
        self.fake_env = FakeEnv(self.model) #FakeEnv to help model unrolling

        if self.load_model != "":
            policy_file = self.file_name if self.load_model == "default" else self.load_model
            self.policy.load(f"./models/{self.experiment_name}/{policy_file}")
            print(f"Model Loaded! From ./models/{self.experiment_name}/{policy_file}")

        self.replay_buffer_Env = ReplayBuffer(state_dim, action_dim)
        self.replay_buffer_Model = ReplayBuffer(state_dim, action_dim)


    def get_action_policy(self, state):
        '''
            Adds exploration noise to an action returned by the TD3 actor.
        '''
        action = (
            self.policy.select_action(np.array(state))
            + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        ).clip(-self.max_action, self.max_action)
        return action

    def get_action_policy_batch(self, state):
        '''
            Adds exploration noise to a batch of actions returned by the TD3 actor.
        '''
        assert len(state.shape) == 2 and state.shape[1] == self.state_dim
        action = (
            self.policy.select_action_batch(np.array(state))
            + np.random.normal(0, self.max_action * self.expl_noise,
                               size=(state.shape[0], self.action_dim))
        ).clip(-self.max_action, self.max_action)
        # Numpy array!
        return action.astype(np.float32)

    def model_rollout(self):
        '''
            This function performs the model-rollout in batch mode for MBPO.
            This rollout is performed once per epoch, and we sample B * M rollouts.
            First, sample B * M transitions from the real environment replay buffer.
            We get B * M states from these transitions.
            Next, predict the action with exploration noise at these states using the TD3 actor.
            Then, use the step() function in FakeEnv to get the next state, reward and done signal.
            Add the new transitions from model to the model replay buffer.
            Continue until you rollout k steps for each of your B * M starting states, or you
            reached episode end for all starting states.

            Note: this implementation only supports k = 1
        '''
        rollout_batch_size = self.rollout_batch_size
        print('[ Model Rollout ] Starting  Rollout length: {} | Batch size: {}'.format(
            self.rollout_horizon, rollout_batch_size
        ))
        unit_batch_size = self.model_rollout_batch_size # B

        batch_pass = self.num_rollouts_per_step # M

        # populate this variable with total number of model transitions collected
        total_steps = 0

        self.replay_buffer_Env.shuffle()

        for j in range(batch_pass):
            if j == batch_pass - 1:
                if rollout_batch_size % unit_batch_size != 0:
                    unit_batch_size = rollout_batch_size % unit_batch_size

            # hint: make use of self.fake_env. Checkout documentation for FakeEnv.py
            # raise NotImplementedError

            # sample from env reply buffer
            start_state, start_action, start_next_state, start_reward, start_not_done = self.replay_buffer_Env.sample(unit_batch_size)
            start_next_state = tf.convert_to_tensor(start_next_state)

            # get action from policy
            policy_action = self.get_action_policy_batch(start_next_state)

            # perform k = 1 step rollout
            fake_env_next_x, fake_env_rewards, fake_env_dones = self.fake_env.step(start_next_state, policy_action)

            # add to replay buffer
            self.replay_buffer_Model.add_batch(start_next_state.numpy(), policy_action, fake_env_next_x.numpy(), fake_env_rewards.numpy(), fake_env_dones.numpy())

        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e})'.format(
            total_steps, self.replay_buffer_Model.size, self.replay_buffer_Model.max_size
        ))

        self.fake_env_steps += total_steps

    def prepare_mixed_batch(self):
        '''
            TODO: implement the mixed batch for MBPO
            Prepare a mixed batch of state, action, next_state, reward and not_done for TD3.
            This function should output 5 tf tensors:
            state, shape (self.batch_size, state_dim)
            action, shape (self.batch_size, action_dim)
            next_state, shape (self.batch_size, state_dim)
            reward, shape (self.batch_size, 1)
            not_done, shape (self.batch_size, 1)
            If MBPO is enabled, each of the 5 tensors should a mixture of samples from the
            real environment replay buffer and model replay buffer. Percentage of samples
            from real environment should match self.percentage_real_transition
            If MBPO is disabled, then simply sample a batch from real environment replay buffer.
        '''
        if self.enable_MBPO:
            # raise NotImplementedError
            env_sample_num = int(self.batch_size * self.percentage_real_transition)
            model_sample_num = self.batch_size - env_sample_num

            states_env, actions_env, next_states_env, rewards_env, not_dones_env = self.replay_buffer_Env.sample(env_sample_num)
            states_model, actions_model, next_states_model, rewards_model, not_dones_model = self.replay_buffer_Model.sample(env_sample_num)

            states = tf.concat([states_env, states_model], axis=0)
            actions = tf.concat([actions_env, actions_model], axis=0)
            next_states = tf.concat([next_states_env, next_states_model], axis=0)
            rewards = tf.concat([rewards_env, rewards_model], axis=0)
            not_dones = tf.concat([not_dones_env, not_dones_model], axis=0)

            return states, actions, next_states, rewards, not_dones
        
        else:
            # TD3 setting, return a sample batch from real environment relay buffer
            return self.replay_buffer_Env.sample(self.batch_size)

    def plot_training_curves(self, evaluations, evaluate_episodes, evaluate_timesteps):
        '''
            Plotting script. You should include these plots in the writeup.
            Do not modify.
        '''
        evaluations_reward = [i[0] for i in evaluations]
        evaluations_error = [i[1] for i in evaluations]
        evaluations_origin = [i[2] for i in evaluations]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(evaluate_timesteps, evaluations_reward)
        ax1.set_xlabel("Training Timesteps")
        ax1.set_ylabel("Evaluation Reward")
        ax1.set_title("Reward vs Training Timesteps")
        ax2.plot(evaluate_timesteps, evaluations_error)
        ax2.plot(evaluate_timesteps, evaluations_origin)
        ax2.legend(["Final Error", "Original Error"])
        ax2.set_xlabel("Training Timesteps")
        ax2.set_ylabel("Evaluation Error")
        ax2.set_title("Error vs Training Timesteps")
        if self.enable_MBPO:
            algo_str = "MBPO"
        else:
            algo_str = "TD3"
        fig.suptitle("Training Curves for " + algo_str, fontsize=20)
        fig.savefig("./results/training_curve_{}_{}.png".format(algo_str, self.experiment_name))

    def train(self):
        '''
            Main training loop for both TD3 and MBPO. See Figure 2 in writeup.
        '''
        self.init_models_and_buffer()
        
        # make environment
        seed_coeff, seed_init = self.seed_coeff, self.seed_init
        env = sim.make(self.env_name, seed_coeff, seed_init, goal_condition=self.HER, random_init_target=self.random_init_everytime)
        # Evaluate untrained policy
        if self.load_model != "":
            evaluations = list(np.load(f"results/{self.experiment_name}/{self.file_name}.npy", allow_pickle=True))
            evaluate_timesteps = [i * self.eval_freq for i in range(len(evaluations))]
            evaluate_episodes = [0]
            start_t = evaluate_timesteps[-1]

        else:
            evaluations = [self.eval_policy()]

            evaluate_timesteps = [0]
            evaluate_episodes = [0]
            start_t = 0

        state, done = env.reset(), False

        # Set episode_reward appropriately
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # perpare for goal conditional RL if enabled
        if self.HER:
            action_traj = []
            state_traj = [state[:env.state_space]]

        for t in range(start_t, int(self.max_timesteps)):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.start_timesteps:
                action = env.sample_action()
            else:
                if t == self.start_timesteps:
                    print("Start get action from policy network")
                action = self.get_action_policy(state)
            
            if self.HER:
                # record action trajactory
                action_traj.append(action)

            # Perform model rollout and model training at appropriate timesteps 
            if not self.enable_MBPO:
                # Perform action
                next_state, reward, done, info = env.step(action) # state is conditioned on target state if in HER mode
                if self.HER:
                    state_traj.append(next_state[:env.state_space]) # save the state trajactory
                print("Infor for step {}: {}; action: {}".format(t, info, np.abs(action).max()))
                episode_reward += reward
                # Store data in replay buffer
                # print("action", type(action))
                self.replay_buffer_Env.add(state, action, next_state, reward, done)

                state = next_state
                
                # Train agent after collecting sufficient data
                if t > self.start_timesteps:
                    state_batch, action_batch, next_state_batch, reward_batch, not_done_batch = self.prepare_mixed_batch()

                    # Perform multiple gradient steps per environment step for MBPO
                    self.policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)

            else:
                # Perform action
                next_state, reward, done, info = env.step(action)

                episode_reward += reward

                # Store data in replay buffer
                self.replay_buffer_Env.add(state, action, next_state, reward, done)

                state = next_state

                if t % self.model_update_freq == 0:
                    # train dynamic model
                    self.model.train(self.replay_buffer_Env)
                    
                    # Train agent after collecting sufficient data
                    self.model_rollout()

                # Perform multiple gradient steps per environment step for MBPO
                if t > self.start_timesteps:
                    for g in range(self.num_gradient_updates):
                        state_batch, action_batch, next_state_batch, reward_batch, not_done_batch = self.prepare_mixed_batch()
                        self.policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)
                    
                # raise NotImplementedError
            
            if done:
                # perform HER
                if self.HER:
                    # using future state as goal
                    # state_traj, action_traj
                    for time_step in range(len(action_traj) - 1):
                        action_t = action_traj[time_step]
                        state_t = state_traj[time_step]
                        # print(f"time_step: {time_step}; len(action_tra) {len(action_traj)}; len(state_traj) {len(state_traj)}")
                        next_state_t = state_traj[time_step+1]
                        # sample goals:
                        goals_indx = np.random.randint(time_step+1, len(state_traj), size=(self.HER_k,))
                        for j in goals_indx:
                            goal = state_traj[j]
                            reward_, done_, info_, distance_to_target_ = env.get_reward(next_state_t, time_step+1, goal) # t=time_step+1 to calculate done
                            # condition on goal
                            print("Reward -------------------------------- {}".format(reward_))
                            state_goal = np.concatenate((state_t, goal))
                            next_state_goal = np.concatenate((next_state_t, goal))
                            self.replay_buffer_Env.add(state_goal, action_t, next_state_goal, reward_, done_)
                    # clear the traj
                    state_traj, action_traj = [], []
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                eval_results = self.eval_policy()
                evaluations.append(eval_results)
                evaluate_episodes.append(episode_num+1)
                evaluate_timesteps.append(t+1)
                if len(evaluations) > 1:
                    self.plot_training_curves(evaluations, evaluate_episodes, evaluate_timesteps)
                np.save(f"./results/{self.experiment_name}/{self.file_name}", evaluations)
                if self.save_model:
                    self.policy.save(f"./models/{self.experiment_name}/{self.file_name}")
           
            
