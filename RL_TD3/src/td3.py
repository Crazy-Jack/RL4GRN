'''
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Adopted from author's PyTorch Implementation
'''
# pylint: disable=C0103, R0913, R0901, W0221, R0902, R0914
import copy
import tensorflow as tf
from tensorflow import keras

class Actor(keras.Model):
    '''
        The actor in TD3. Architecture from authors of TD3
    '''
    def __init__(self, action_dim, max_action):
        super().__init__()

        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(256, activation="relu")
        self.l4 = keras.layers.Dense(256, activation="relu")
        self.l5 = keras.layers.Dense(action_dim)
        self.max_action = max_action


    def call(self, state):
        '''
            Returns the tanh normalized action
            Ensures that output <= self.max_action
        '''
        a = self.l1(state)
        a = self.l2(a)
        a = self.l3(a)
        a = self.l4(a)
        # a = self.l5(a)
        return self.max_action * keras.activations.tanh(self.l5(a))


class Critic(keras.Model):
    '''
        The critics in TD3. Architecture from authors of TD3
        We organize both critics within the same keras.Model
    '''
    def __init__(self):
        super().__init__()

        # Q1 architecture
        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(256, activation="relu")
        self.l4 = keras.layers.Dense(256, activation="relu")
        self.l5 = keras.layers.Dense(1)

        # Q2 architecture
        self.l6 = keras.layers.Dense(256, activation="relu")
        self.l7 = keras.layers.Dense(256, activation="relu")
        self.l8 = keras.layers.Dense(256, activation="relu")
        self.l9 = keras.layers.Dense(256, activation="relu")
        self.l10 = keras.layers.Dense(1)


    def call(self, state, action):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        q1 = self.l4(q1)
        q1 = self.l5(q1)

        q2 = self.l6(sa)
        q2 = self.l7(q2)
        q2 = self.l8(q2)
        q2 = self.l9(q2)
        q2 = self.l10(q2)
        return q1, q2 # (bz, 1), (bz, 1)


    def Q1(self, state, action):
        '''
            Returns the output for only critic 1. Used to compute actor loss.
        '''
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        q1 = self.l4(q1)
        q1 = self.l5(q1)

        return q1


class TD3():
    '''
        The TD3 main class. Wraps around both the actor and critic, and provides
        three public methods:
        train_on_batch, which trains both the actor and critic on a batch of
        transitions
        select_action, which outputs the action by actor given a single state
        select_action_batch, which outputs the actions by actor given a batch
        of states.
    '''
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.state_dim = state_dim
        self.actor = Actor(action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        self.actor.compile(optimizer=self.actor_optimizer)
        self.critic.compile(optimizer=self.critic_optimizer)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        '''
            Select action for a single state.
            state: np.array, size (state_dim, )
            output: np.array, size (action_dim, )
        '''
        state = tf.convert_to_tensor(state.reshape(1, -1))
        return self.actor(state).numpy().flatten()

    def select_action_batch(self, state):
        '''
            Select action for a batch of states.
            state: np.array, size (batch_size, state_dim)
            output: np.array, size (batch_size, action_dim)
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        return self.actor(state).numpy()


    def train_on_batch(self, state, action, next_state, reward, not_done):
        '''
            Trains both the actor and the critics on a batch of transitions.
            state: tf tensor, size (batch_size, state_dim)
            action: tf tensor, size (batch_size, action_dim)
            next_state: tf tensor, size (batch_size, state_dim)
            reward: tf tensor, size (batch_size, 1)
            not_done: tf tensor, size (batch_size, 1)
            You need to implement part of this function.
        '''
        self.total_it += 1
        batch_size = state.shape[0]

        # Select action according to policy and add clipped noise
        noise = tf.clip_by_value(tf.random.normal(action.shape) * self.policy_noise,
                                 -self.noise_clip, self.noise_clip)

        next_action = tf.clip_by_value(self.actor_target(next_state) + noise,
                                       -self.max_action, self.max_action) # (bz, action_dim)
        
        
        Q_value_1, Q_value_2 = self.critic_target(next_state, next_action)

        min_q_value = tf.math.minimum(tf.reshape(Q_value_1, [batch_size, 1]), tf.reshape(Q_value_2, [batch_size, 1])) # (bz, 1)

        target_q = reward + self.discount * min_q_value * not_done # (bz, 1)

        with tf.GradientTape() as critic_tape:
            
            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action) # [(bz, 1), (bz, 1)]

            # Compute critic loss
            loss_critic1 = tf.reduce_mean((target_q - current_q1) ** 2)
            loss_critic2 = tf.reduce_mean((target_q - current_q2) ** 2)

            loss = loss_critic1 + loss_critic2
            with critic_tape.stop_recording():
                logstr = "q1 {:.5f}; q2 {:.5f}; min_q_value {:.3f}; closs {:.10f}; closs1 {:.10f}; closs2 {:.10f} ".format(Q_value_1.numpy().mean(), \
                                                                                Q_value_2.numpy().mean(), min_q_value.numpy().mean(), loss.numpy(), loss_critic1.numpy(), loss_critic2.numpy())


        # Optimize the critic
        # print(self.critic.trainable_variables)
        critic_grad = critic_tape.gradient(loss, self.critic.trainable_variables)

        # Delayed policy updates
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Update actor 
        if self.total_it % self.policy_freq == 0:
            # Compute actor losses
            with tf.GradientTape() as actor_tape:
                next_action = self.actor(state)
                actor_loss = - tf.math.reduce_mean(self.critic.Q1(state, next_action))

            # Update actor
            actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            # print(self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            logstr += "act_loss {:.10f}".format(actor_loss.numpy())
            # print(logstr)
            # Update the frozen target models
            
            ## target actor 
            actor_ws = []
            for i, w in enumerate(self.actor.weights):
                actor_ws.append(self.tau * copy.deepcopy(w) + (1.0 - self.tau) * copy.deepcopy(self.actor_target.weights[i]))
            self.actor_target.set_weights(actor_ws)

            ## target critic
            critic_ws = []
            for i, w in enumerate(self.critic.weights):
                critic_ws.append(self.tau * copy.deepcopy(w) + (1.0 - self.tau) * copy.deepcopy(self.critic_target.weights[i]))
            self.critic_target.set_weights(critic_ws)




    def save(self, filename):
        '''
            Saves current weight of actor and critic. You may use this function for debugging.
            Do not modify.
        '''
        self.critic.save_weights(filename + "_critic")
        self.actor.save_weights(filename + "_actor")


    def load(self, filename):
        '''
            Loads current weight of actor and critic. Notice that we initialize the targets to
            be identical to the on-policy weights.
        '''
        self.critic.load_weights(filename + "_critic")
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_weights(filename + "_actor")
        self.actor_target = copy.deepcopy(self.actor)
