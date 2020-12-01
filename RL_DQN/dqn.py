import numpy as np, gym, sys, copy, argparse
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential,Model
import random
import tensorflow as tf
from collections import deque
from pathlib import Path
import keras
from keras import backend as K_back
import pickle
import os
import matplotlib.pyplot as plt

env = make('random_generate', seed_network=0, seed_init=1)

class QNetwork():
    def __init__(self, env):
        self.learning_rate=0.01
        self.obs_space=env.state_space #self.obs_space=env.observation_space.shape[0]
        self.ac_space=env.action_space  #self.ac_space=env.action_space.n        
        print("Building DQN model")
        self.model=self.build_model_DQN()

    def save_model_weights(self, name):
        self.model.save(name)# Helper function to save your model / weights. 

    def load_model(self, model_file):
        self.model=keras.models.load_model(model_file, custom_objects={"K_back": K_back}) # Helper function to load an existing model.

    def load_model_weights(self,weight_file):
        self.model.load_weights(weight_file) # Helper funciton to load model weights. 

    def build_model_DQN(self): #Builds a DQN
        model=Sequential()
        model.add(Dense(units=24,input_dim=self.obs_space,activation='relu',
                        kernel_initializer='he_uniform'))
        #model.add(Dense(units=24,activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(units=self.ac_space,activation='linear',kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
        return model
    
EPISODES=500 #NUMBER OF EPISODES 

class Replay_Memory():

    def __init__(self, memory_size=500, burn_in=100):
        self.burn_in=burn_in
        self.memory=memory_size
        self.mem_queue=deque(maxlen=self.memory)

    def sample_batch(self, batch_size=32):
        return random.sample(self.mem_queue,batch_size)

    def append(self, transition):   
        if(len(self.mem_queue)<self.memory):
            self.mem_queue.append(transition)
        else:
            self.mem_queue.popleft()
            self.mem_queue.append(transition)

class DQN_Agent():
    def __init__(self, env):

        self.net=QNetwork(env)
        self.obs_space=env.state_space #self.obs_space=env.observation_space.shape[0]
        self.ac_space=env.action_space  #self.ac_space=env.action_space.n    
        ######################Hyperparameters###########################
        self.env=env
        self.epsilon=0.5
        self.epsilon_min=0.05
        self.epsilon_decay=0.99
        self.gamma=0.99
        self.max_itr=1000000
        self.batch_size=32
        #self.max_reward=100 #Used for saving a model with a reward above a certain threshold
        self.memory_queue=Replay_Memory(memory_size=500, burn_in=100)
        ###############################################################
        self.avg_rew_buffer=10
        self.avg_rew_queue=deque(maxlen=self.avg_rew_buffer)
        self.model_save=50
        self.test_model_interval=10

        
    def epsilon_greedy_policy(self, q_values,epsi):# Creating epsilon greedy probabilities to sample from.  
        if random.uniform(0,1)<=epsi:
            return random.randint(0,self.ac_space-1) #Q-Values shape is batch_size x ac
        else:
            return np.argmax(q_values[0])

    def greedy_policy(self, q_values):# Creating greedy policy for test time.
        return np.argmax(q_values[0])

    def train(self):
        testX,testY,testZ=[],[],[]
        batch_size,max_,avg_rew_test,itr=self.batch_size,0,0,0

        print("Using Experience Replay") #Burn In 
        self.burn_in_memory()

        for epi in range(EPISODES):
            state=np.reshape(self.env.reset(),[1,self.obs_space])#Reset the state
            total_rew=0
            start_time = time.time()

            
            while True:
                itr+=1
                ac=self.epsilon_greedy_policy(self.net.model.predict(state),self.epsilon)   #get action by e-greedy
                
                n_s,rew,is_t, _ = self.env.step(ac) #Find out next state and rew for current action

                n_s=np.reshape(n_s,[1,self.obs_space])   #Append to queue
                self.memory_queue.append([state,ac,rew,is_t,n_s])
                batch=self.memory_queue.sample_batch(batch_size=batch_size) #Get samples of size batch_size

                #Create array of states and next states
                batch_states=np.zeros((len(batch),self.obs_space))
                batch_next_states=np.zeros((len(batch),self.obs_space))
                actions,rewards,terminals=[],[],[]

                for i in range(0,len(batch)):
                    b_state, b_ac, b_rew, b_is_t, b_ns=batch[i] #Returns already reshaped b_state and b_ns
                    batch_states[i]=b_state
                    batch_next_states[i]=b_ns
                    actions.append(b_ac)
                    rewards.append(b_rew)
                    terminals.append(b_is_t)

                #Get Predictions
                batch_q_values=self.net.model.predict(batch_states)
                batch_next_q_values=self.net.model.predict(batch_next_states)
                
                for i in range(0,len(batch)):
                    if terminals[i]: #Corresponds to is_terminal in sampled batch
                        batch_q_values[i][actions[i]]=rewards[i]

                    else: #If not
                        batch_q_values[i][actions[i]]=rewards[i]+self.gamma*(np.amax(batch_next_q_values[i]))  
                #Perform one step of SGD
                self.net.model.fit(batch_states,batch_q_values,batch_size=batch_size,epochs=1,verbose=0)
                self.epsilon*=self.epsilon_decay
                self.epsilon=max(self.epsilon,self.epsilon_min)
                total_rew+=rew
                state=n_s
                
                if is_t:
                    break
            
            #test model at intervals
            if((epi+1)%self.test_model_interval==0):
                testX.append(epi)
                avg_rew_test, td_error=self.test()
                testY.append(avg_rew_test)
                testZ.append(td_error)

            #Remove and add rewards to calculate avg reward
            if(len(self.avg_rew_queue)>self.avg_rew_buffer):
                self.avg.rew_queue.popleft()
            self.avg_rew_queue.append(total_rew)
            avg_rew=sum(self.avg_rew_queue)/len(self.avg_rew_queue)
            

            end_time = time.time()
            time_delta_step = end_time - start_time
            distance_to_target = ((n_s - env.target_state) ** 2).sum()
            print("distance_to_target: {}".format(distance_to_target))
            print("episode={}; iteration={}; reward={};time={:.4f} s".format(epi,itr,avg_rew,time_delta_step))



        fig=plot_eval(testX,testY,testZ) #Plotting after episodes are done
        fig.savefig("plot.jpeg")
            

    def test(self):
        test_episodes=5
        rewards=[]
        td_errors=[]
        for e in range(test_episodes):
            state = np.reshape(self.env.reset(),[1,self.obs_space])
            time_steps = 0
            total_reward_per_episode = 0
            td_error_per_episode=[]
            while True:
                #if(self.render):
                    #self.env.render()
                action = self.epsilon_greedy_policy(self.net.model.predict(state),0.05)
                next_state, reward, is_t, _ = self.env.step(action)
                next_state=np.reshape(next_state,[1,self.obs_space])
                td_error=abs(reward + self.gamma *self.net.model.predict(next_state)-self.net.model.predict(state))
                state = next_state
                total_reward_per_episode+=reward
                td_error_per_episode.append(td_error)
                time_steps+=1
                if is_t:
                    break
            rewards.append(total_reward_per_episode)
            td_errors.append(np.mean(td_error_per_episode))
            

        avg_rewards_=np.mean(np.array(rewards))
        td_error_=np.mean(np.array(td_errors))
        std_dev=np.std(rewards)
        print("AvgRew={},Std={}".format(avg_rewards_,std_dev))
        print("td_error={}".format(td_error_))
        return avg_rewards_, td_error_

    def burn_in_memory(self):
        # Initialize replay memory with a burn_in number of episodes / transitions. 
        memory_size=0
        state=np.reshape(self.env.reset(),[1,self.obs_space])
        
        while(memory_size<self.memory_queue.burn_in):
            ac=random.randint(0,self.ac_space-1)
            n_s,rew,is_t,_=self.env.step(ac)
            n_s=np.reshape(n_s,[1,self.obs_space])
            
            transition=[state,ac,rew,is_t,n_s]
            self.memory_queue.append(transition)
            state=n_s
            if is_t:
                state=np.reshape(self.env.reset(),[1,self.obs_space])
            memory_size+=1

        print("Burned Memory Queue")


def plot_eval(testX, testY, testZ):
  fig=plt.figure(figsize=(5.5, 10))
  plt.subplot(211)
  plt.title("Average award")
  plt.xlabel("Training Episode")
  plt.ylabel("Average Test Reward for 10 episodes")
  plt.plot(testX, testY)

  plt.subplot(212)
  plt.title("TD error")
  plt.plot(testX, testZ)
  plt.xlabel("Training Episode")
  plt.ylabel("TD error for 10 episodes")
  plt.show()
        
  return fig

def main(args):
    env = make('random_generate', seed_network=0, seed_init=1)
    model = DQN_Agent(env)
    model.train()
  

if __name__ == '__main__':
    main(sys.argv)