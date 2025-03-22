#!/usr/bin/env python
# coding: utf-8

# In[1]:


####!/apps/anaconda3/bin/python

from random import choices 
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
import os
from scipy import optimize
import datetime
from joblib import Parallel, delayed
import joblib
import csv   
from time import gmtime, strftime
from sklearn.neighbors import KernelDensity
from itertools import product
import ast
from scipy import integrate
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import scipy.stats as stats
import math
from collections import deque
from os.path import isfile, join
from os import listdir
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import pylab as py
import seaborn as sns  
import statistics
# In[45]:



### code adapted from https://github.com/cyoon1729/deep-Q-networks
class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim,nn_dim = 3):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], nn_dim),
            nn.ReLU(),
            nn.Linear(nn_dim, nn_dim),
            nn.ReLU(),
            nn.Linear(nn_dim, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[start]
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=0.01, gamma=0.999999, nn_dim = 3,loss_type = 'MSE', buffer_size=1000
                 ,opt_method = 'Adam', weight_decay_val = 0, momentum_val = 0):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_list = []
        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, env.action_space.n,nn_dim).to(self.device)

        if opt_method == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate,weight_decay = weight_decay_val)
        elif opt_method == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=learning_rate,weight_decay = weight_decay_val,momentum = momentum_val)
    
        if loss_type == 'MSE':
            self.loss_func = nn.MSELoss()
        elif loss_type == 'HuberLoss':
            self.loss_func = nn.HuberLoss(delta=1.0)
        else:
            pass

    def get_action(self, state, eps=0.05):
        
        middle_index = int(len(state)/2)
        queue_length_vector = state[middle_index:]
        age_vector = state[:middle_index]
        
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < eps):
             
            return self.env.action_space.sample()
        
        
        if np.max(queue_length_vector) >= 10:
            action = np.argmax(age_vector) #if queue length too large, use FIFO rule
            pass
        
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.loss_func(curr_Q, expected_Q)
        return loss

    def get_model(self): #return the model we need
        return self.model

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        #print('batch')
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        
        loss_val=loss.cpu().detach().numpy()
        self.loss_list.append(float(loss_val)) #append loss value
    
        self.optimizer.step()   
        
        #for name, param in self.model.named_parameters():
            #print(name, param.grad)
                


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, reset_paras,eps_val = 0.05):

    episode_rewards = []
    
    cost_vector = reset_paras['cost_vector']
    arrival_distribution = reset_paras['arrival_distribution']
    service_distribution = reset_paras['service_distribution']
    mixture_prob_arrival = reset_paras['mixture_prob_arrival']
    mixture_prob_service = reset_paras['mixture_prob_service']
    reward_type = reset_paras['reward_type']
    
    
    exp_episodes = max_episodes/2 # do exploration for the first half time
    
    for episode in range(max_episodes):
        state = env.reset(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service,reward_type)
        episode_reward = 0
        all_t_rewards = []

        for step in range(max_steps):
            if episode <= exp_episodes:
                action = agent.get_action(state, 1) #pure explore..
            else:
                action = agent.get_action(state, eps_val)  
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            #print('state',state,'action',action,'reward', reward, 'next_state',next_state, done)
            episode_reward += reward
            all_t_rewards.append(reward)

            if episode >= exp_episodes+1 and len(agent.replay_buffer) > batch_size: #start updating after some episodes
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                #print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
   
    return (agent.get_model())

## function returns the large deviation rate or robustness estimator for input y, given the random samples x
def robustness_estimator_grad(Lambda,y,x):
    '''
    x is the input random observations, Lambda is the parameter we want to find, y is the thresohld
    '''
    exp_labmda_x = np.exp([Lambda*_ for _ in x]) 
    grad = y - np.mean(np.multiply(x,exp_labmda_x)) / np.mean(exp_labmda_x) #formula found in A.5.1 Proof of Lemma 
    return (grad)
 

def random_exponential(rate): # mean = 1/rate
    return np.random.exponential(1/rate) #define this function to make the parameter be the rate, i.e. 1/mean


def random_half_normal(rate): # mean = 1/rate, half normal distribution
    x = math.sqrt(math.pi/2) * abs(np.random.normal(0,1/rate))
    return x

def identity(rate): ##define the identity function, no randomness, just for testing
    return 1/rate
 
###prepare  for non-stationarity  
 
def customized_function_generator(func,x0,para=[]): #return a function, can be used as rate or mixture
    if func == 'exponential':
        def func(x):
            return x0+np.exp(-para*x)
    elif func == 'constant':
        def func(x):
            return x0
    elif func == 'truncated_linear': #gradually linearly increases to a value
        def func(x):
            k = para['slope']
            u = para['upper_limit']
            return min(u,x0+k*x)
    elif func == 'truncated_linear_decrease': #gradually linearly decreases to a value
        def func(x):
            k = para['slope']
            d = para['lower_limit']
            return max(d,x0-k*x)
    elif func == 'sin': #sin function to reflect high violatility
        def func(x):
            freq = para['freq']
            return (max(x0,x0 + para['range'] * math.sin(freq*x)))
    elif func == 'step_func': #step function, suddenly change value
        def func(x):
            b = para['break_point']
            x1 = para['new_value']
            if x > b:
                return x1
            else:
                return x0
    return func 

def rv_generator(base_distribution, mixture_prob, t): #base dist:base of distribution, mixture_prob: individual dis, t:time
    num_distr = len(base_distribution)

    mixture_prob = [item(t) for item in mixture_prob]  #this is a function of time
    mixture_prob /= np.sum(mixture_prob)  #normalize, in case it doesn't sum up to 1
    mixture_rv = np.random.choice(np.arange(num_distr), size=1, p = mixture_prob)[0]
    distr = base_distribution[mixture_rv]

    rv_para = distr["kwargs"].copy() #use copy to avoid modifying the original value
    for key, value in rv_para.items():
        rv_para[key] = value(t) #convert parameters to a value from function t
    try:
        rv = distr["type"](**rv_para)[0]
    except:
        rv = distr["type"](**rv_para)
    return rv 

def calculate_service_rate(base_distribution, mixture_prob, t): #output  mu=1/E[.] for service, assume for default dist paras are in rate form (o/w create a dict with key 'rate'), so it's easy to calculate the overall rate based on the individual rate
    num_distr = len(base_distribution)
    mixture_prob = [item(t) for item in mixture_prob]  #this is a function of time
    mixture_prob /= np.sum(mixture_prob)  #normalize, in case it doesn't sum up to 1

    mean_list = [] #record mean of each indivial r.v.

    for distr in base_distribution:
        rv_para = distr["kwargs"].copy() #use copy to avoid modifying the original value
        for key, value in rv_para.items():
            rv_para[key] = value(t) #convert parameters to a value from function t

        #print(list(distr['kwargs'].values()))
        try:
            mean_list.append(1/list(rv_para.values())[0])
        except:
            mean_list.append(1/list(distr["rate"].values())[0])

    overall_avg = np.inner(mean_list, mixture_prob)
    return (1/overall_avg) #this is the overall rate for this distribution
 
def get_fist_cust_age_for_all_queue(x_list): #pull out the age of the first cust for all queues, -1 indicates empty queue
    return ([x_list[i][0] if len(x_list[i]) > 0 else -1 for i in range(len(x_list))])


def update_cust_age(x,c): #input x(list of lists), output each element of x+=c, only update non-empty elements, empty \neq 0 in this case!
    for j in range(len(x)): #loop over all lists
        list_of_x = x[j]
        if len (list_of_x) > 0: #if there is an element
            x[j] = [i+c for i in list_of_x]
    return x


class Queueing_Process(Env):  ## define the Queueing process 
    def __init__(self, cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,reward_type): 
        N = len(cost_vector) #N number of queues  
        self.action_space = Discrete(N) # Action which server to serve
        self.observation_space = MultiDiscrete([max_age for _ in range(2*N)]) #set of age (the cust in the queue, and the queue length for each queue we can observe 
     
 
    def step(self, action):

        def update_reward(): #get update of reward, need this because we may want to update 
            self.age_list_all = [a+b for (a,b) in zip(self.age_list,self.age_list_leaving)]  #put age info of cust both in systems and not
            reward = -np.sum([np.sum([cost*(max(age,0)**2) for age in item]) for (item,cost) in zip(self.age_list_all, self.cost_vector)]) #calculate sum c_i age_i^2 for all cust in the system now
            return reward 
        
        def update_state(time_before, time): #use this to get state for arrivals before service completion time
            queue_length_vector = [len(item) for item in self.age_list]
            cum_reward = update_reward() ## this is the cum reward, i.e. what we need to evaluate the model, which is different to reward generated per time step, for that we only need it at times where an action is needed
            first_age_vector = get_fist_cust_age_for_all_queue(self.age_list) #we can only observe the age of the first cust, to simplify the policy, obs=age for first cust, empty = -1 so policy won't select this
            observation = first_age_vector + queue_length_vector ## put queue_length_vector at last, this is the ob space
            return {'previous_system_time': time_before,'system_time':time, 'queue_length_vector':queue_length_vector, 'cum_reward':cum_reward
                   ,'first_age_vector':first_age_vector, 'observation':observation} #record at that arrival time, the queue length and cum reward

        #dynamics: first decide who to serve, then new custs arrive Q[t]=queue length at time t, before departure
        self.previous_system_time = self.system_time #this records the system when the step starts.
        self.service_rv_time = rv_generator(base_distribution = self.service_distribution[action], mixture_prob = self.mixture_prob_service, t = self.system_time)   # this is the time of service until sucess
        self.if_in_service = sum([len(_) for _ in self.age_list]) > 0 #1 anyone in the queue
        N = self.N 
        
        previous_reward = self.cum_reward ##save this val
        
        self.previous_age_first_list = get_fist_cust_age_for_all_queue(self.age_list) #record the previous 1st age vector
        self.previous_queue_length_vector = [len(item) for item in self.age_list]
        
        inter_service_info = [] #used to record state value changes before service finishes
                
        if self.if_in_service == 1: #if the system is running:
            age_list_copy = self.age_list.copy() #need this, so we can update the age to age+service time, before finishing the service more easily

            time_new_arrival = [0 for _ in range(N)] #new arrival custs' next arrival_time only records 1 value per queue
            if_finish_new_arrival = [0 for _ in range(N)] # if finished, for new arrivals

            cum_time = 0 # if cum_time >=  service time, then stop all queues
             #the following while is to capture the arrival before the service is completed
            while np.sum(if_finish_new_arrival) < N: #when the simulation process is not finished
                for i in range(N):
                    if if_finish_new_arrival[i] == 0: #only simulate if this queue is not finished
                        arrival_time_from_last_step = self.failed_cust_arrival_time[i] #obtained from last step, remaining next arrival time
                        #simulate the next arrival
                        if arrival_time_from_last_step > 0: #need strictly larger than 0
                            
                            if (cum_time + arrival_time_from_last_step) > self.service_rv_time: ##e.g. 1st service time=5, rand_arrival=100, after 1st iter, fail_cust=95, 2nd service=10, failed cust <- 95-10=85
                                if_finish_new_arrival[i] = 1 # in this case,  finish
                                
                            time_new_arrival[i] = (arrival_time_from_last_step) 
                                
                                 
                        else:
                            temp_val = rv_generator(base_distribution = self.arrival_distribution[i], mixture_prob = self.mixture_prob_arrival, t = self.system_time) #the first cust if no arrival before service complete, this might be overwritten by arrival_time_from_last_step
                            if (cum_time + temp_val) >  self.service_rv_time: 
                                if_finish_new_arrival[i] = 1  # in this case,  finish
                                
                            time_new_arrival[i] = (temp_val)
                #print('np.sum(if_finish_new_arrival',np.sum(if_finish_new_arrival))
                ## start counting age from this part
                if np.sum(if_finish_new_arrival) == N: #if all arrivals are here
                    self.age_list = update_cust_age(self.age_list,self.service_rv_time - cum_time) #e.g. last arrival in this cycle is  time 10, service time is 15, then everyone should increase age by 5
                    self.system_time = self.system_time + self.service_rv_time - cum_time #update system_time as well
                    break #finishes the loop
                
                ## update failed_cust_arrival_time and cum_time, and reward
                else:
                    #print('time_new_arrival',time_new_arrival) 
                    min_arrival_time = np.min(time_new_arrival) #this is the time we have to add to the system time
                    min_arrival_index = np.argmin(time_new_arrival) #this is the index of the first newly arrived cust 
                    system_time_copy = self.system_time #before updaing, make a copy for inter service use
                    self.system_time = self.system_time + min_arrival_time #update accordingly, use min arrival time, next time if do sampling, use this as time index
                    #now we should update age and inter_service_info
                    self.age_list = update_cust_age(self.age_list,min_arrival_time) #for those still in the system before the service begins, add service_time to all of them
                    self.age_list[min_arrival_index].append(0) #new cust arrives, age 0
                    inter_service_info.append(update_state(system_time_copy,self.system_time)) #update state info, in the  time before service completion
                    
                     
                    cum_time = cum_time + min_arrival_time #update the cum_time, note that it's done outside the for loop
                    self.failed_cust_arrival_time = [x - min_arrival_time for x in time_new_arrival] #use this, so we can simulate in the next iteration
                    self.failed_cust_arrival_time[min_arrival_index] = -1  
                    
                     
                    
        else: # o/w jump to the next time until one of the new cust arrive, in this case, no inter service info needed
            new_arrival_time_list_no_service = [rv_generator(base_distribution = self.arrival_distribution[i], mixture_prob = self.mixture_prob_arrival, t = self.system_time) for i in range(N)]

            min_arrival_time = np.min(new_arrival_time_list_no_service) #this is the time we have to add to the system time
            min_arrival_index = np.argmin(new_arrival_time_list_no_service) #this is the index of the first newly arrived cust #later fix possibility of multiple arrival same time 
            self.system_time = self.system_time + min_arrival_time #update accordingly, use min arrival time
            self.age_list[min_arrival_index].append(0) #the new cust will have age 0 and will be served in the next step, this is how we update age in this case, we don't add service time now since it's happening in the next iteration

            self.failed_cust_arrival_time = [x - min_arrival_time for x in new_arrival_time_list_no_service] #this should be all arrival - min_arrival_time
            self.failed_cust_arrival_time[min_arrival_index] = -1 #indicate that this cust should not be classified as arrival after min_arrival_time happens

           
            
        if self.if_in_service == 1:  
            try:
                leaving_cust_age = age_list_copy[action][0] + self.service_rv_time #records the leaving cust's age, add service time, will be used to calculate the reward
                self.age_list_leaving[action].append(leaving_cust_age) #append this leaving cust's info to the leaving list. 
                self.cust_service_time_list[action].append(self.service_rv_time) 
                del self.age_list[action][0] #this cust leaves, so delete from the list 
            except:
                pass #in this case, the server selects one empty queue, which is bad, we do nothing    
        else:
            leaving_cust_age = 0 #in this case, the age is just 0 no need to append, since does not affect cost
            
        ## finish the entire loop, now compute what we need    
        first_age_vector = get_fist_cust_age_for_all_queue(self.age_list) #we can only observe the age of the first cust, to simplify the policy, obs=age for first cust, empty = -1 so policy won't select this
        queue_length_vector = [len(item) for item in self.age_list]
        observation = first_age_vector + queue_length_vector ## put queue_length_vector at last, this is the ob space

        #cum reward is for evaluation purpose only
        self.cum_reward = update_reward()
        
        
        if self.reward_type == 'c*age':
            reward =  - np.dot(self.cost_vector, first_age_vector)
            #reward =  - np.dot(self.cost_vector, queue_length_vector)
        elif self.reward_type == 'cum_diff':
            reward = self.cum_reward - previous_reward ## diff version
        
        
        if self.previous_queue_length_vector[action] == 0 and np.sum(self.previous_queue_length_vector)>0:#if choose empty age, add large punishment
            reward = reward - 1000000
            pass 
        
        current_service_rate = [calculate_service_rate(base_distribution = self.service_distribution[i], mixture_prob = self.mixture_prob_service, t = self.system_time) for i in range(N)] 

        info = {'action':action,'age_list':self.age_list,'service_rv_time':self.service_rv_time,'cum_reward':self.cum_reward
                ,'if_in_service':self.if_in_service,'system_time': self.system_time,'previous_system_time': self.previous_system_time
                ,'queue_length_vector':queue_length_vector,'age_list_leaving':self.age_list_leaving
                ,'current_service_rate':current_service_rate,'inter_service_info':inter_service_info}
        
    
        done = False  #assume never ends..
        # Return step information
        return observation, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self, cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, reward_type):  ##reset 
        N = len(cost_vector)
        self.N = N
        self.age_list = [[0] for _ in range(N)] #assume each queue starts with 1 cust in the queue for simplicity, since our system will make decisions right away
        self.age_list_leaving = [[] for _ in range(N)]  #record the age of all leaving customers, each type has a list
        self.age_list_all = [[] for _ in range(N)] #record the age of all  customers, each type has a list
        self.cust_service_time_list = [[] for _ in range(N)]  #record the service time of all leaving customers, each type has a list
        self.cum_reward = 0 #use this to track of cum_reward
        
        self.system_time = 0 #this is the global system time that records the tree time instead of the event time
        self.previous_system_time = 0

        self.arrival_distribution = arrival_distribution  #arrival distribution parameter
        self.service_distribution = service_distribution #service distribution parameter
        self.mixture_prob_arrival = mixture_prob_arrival
        self.mixture_prob_service = mixture_prob_service
      
        self.if_in_service = 0 #this records if the system is serving someone
        self.cost_vector = cost_vector
        self.failed_cust_arrival_time = [-1]*N #this is used to record the cust with arrival time >= service time. Due to independence we can't throw those away. -1  means no unarrived custs, in the next time, if this is not empty then we should not do sampling for the first arrival time
        self.previous_age_first_list = []
        self.previous_queue_length_vector = []
        
        first_age_vector = get_fist_cust_age_for_all_queue(self.age_list) #we can only observe the age of the first cust, to simplify the policy, obs=age for first cust
        queue_length_vector = [len(item) for item in self.age_list]
        observation = first_age_vector + queue_length_vector ## put queue_length_vector at last
        
        self.reward_type = reward_type ##add this to try various reward types
        return observation


### this is the policy part, for this part, reward_type is not important, it only matters in the training stage.
def simulate_policy(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,policy,time_len,DRL_model=[],if_access_all_mu=1,reward_type = 'c*age'): 
    env = Queueing_Process(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,reward_type)
    cum_reward_list = []
    N = len(cost_vector)
    queue_length_list = []
    service_rv_time_list = []
    
    previous_system_time_list = []
    system_time_list = []
    action_list = []
    q_vals_list = []
    state_list = []
    
    observation = env.reset(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service,reward_type)
    initial_service_rate = [calculate_service_rate(base_distribution = service_distribution[i], mixture_prob = mixture_prob_service, t = 0) for i in range(N)]
    
    for t in range(time_len):
        env.render()
        if policy == 'c_mu':
            result = np.multiply(cost_vector, observation[:N])  ##only take first N since we are using age vector to make decision now
            try:  
                if if_access_all_mu == 1: #if c_mu rule is allowed to know the mu vector at all time
                    service_rate = info['current_service_rate']
                else:
                    service_rate = initial_service_rate
            except:
                service_rate = initial_service_rate #assume only know the service rate in initial times

            qvals = 0

            result = np.multiply(service_rate, result)
            action = np.argmax(result)  # c_mu rule here, argmax C[i]*age[i]*mu[i]  
        elif policy == 'FIFO': ## FIFO policy selects the one with the largest age
            try:
                action = np.argmax(observation[:N]) ##only take first N since we are using age vector to make decision now
            except:
                action = env.action_space.sample() #all queues empty, just randomly select 1

            qvals = 0

        elif policy == 'DRL': ## use DRL policy in this case
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            middle_index = int(len(observation)/2)
            queue_length_vector = observation[middle_index:]
            age_vector = observation[:middle_index]
            
            state = torch.FloatTensor(observation).float().unsqueeze(0).to(device)
           
            qvals = DRL_model.forward(state) ##use the loaded DRL model here, append this to debug
            action = np.argmax(qvals.cpu().detach().numpy())
            
            if np.max(queue_length_vector) >= 10:
                action = np.argmax(age_vector) #if queue length too large, use FIFO rule
                pass


        q_vals_list.append(qvals)
        action_list.append(action)
        observation, reward, done, info = env.step(action)
        
    
        #print(info['if_in_service'],'service_rv_time:',info['service_rv_time'],'age_list:',info['age_list'],'previous_system_time:',info['previous_system_time'],'system_time:',info['system_time']) 

        
        
        inter_service_info = info['inter_service_info']
        if len(inter_service_info) > 0:
            for item in inter_service_info:
                previous_system_time_list.append(item['previous_system_time'])
                system_time_list.append(item['system_time'])
                cum_reward_list.append(item['cum_reward']) 
                queue_length_list.append(np.sum(item['queue_length_vector'])) #use sum to represent the queue length info to check

            previous_system_time_list.append(inter_service_info[-1]['system_time'])
        else:
            previous_system_time_list.append(info['previous_system_time'])


        service_rv_time_list.append(info['service_rv_time'])
        cum_reward_list.append(info['cum_reward']) 
        queue_length_list.append(np.sum(info['queue_length_vector'])) #use sum to represent the queue length info to check
       
        system_time_list.append(np.sum(info['system_time']))
        state_list.append(observation)

        
        #print ('obs',observation)
    env.close()
    return {'cum_reward':cum_reward_list, 'queue_length': queue_length_list,'service_rv_time_list':service_rv_time_list,'previous_system_time_list':previous_system_time_list,'system_time_list':system_time_list,'action_list':action_list,'q_vals_list':q_vals_list,'state_list':state_list} 



def map_to_all_discrete_time(simulation_result, var, discrete_interval, initial_value = 1): #intput simulation result (queue/reward etc.), map this to a list  discrete_interval for plotting/evaluation purpose at each <= obs time, output the value we care about
    assert len(simulation_result[var]) == len(simulation_result['system_time_list'])
    var_result = [initial_value] + simulation_result[var] #initial x = 1,assumed to be true
    system_time_list = simulation_result['system_time_list']

    var_vector_all_discrete_time = [-1] * len(discrete_interval) #initialize, will overwrite
    system_time_list = [0] + system_time_list #add time 0
    j = 0
    for i in range(len(discrete_interval)):
        eval_time = discrete_interval[i] #we want to extract info at time eval_time
        try:
            while(eval_time >= system_time_list[j]): 
                j = j + 1 #increase j until discrete_interva[i] < system_time_list[j]
        except:
            pass
        j = j -1 #should -1 here since we exceed eval_time
        try:
            var_vector_all_discrete_time[i] = var_result[j]
        except:
            pass
    
    return var_vector_all_discrete_time


#random seed function
def set_seed(seed):
    try:
        pass
        #import tensorflow as tf
        #tf.random.set_seed(seed)
    except Exception as e:
        print("Set seed failed,details are ",e)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ",e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)

def Kernel_Fit(x,band_width=100):
    kde = KernelDensity(bandwidth=band_width, kernel='gaussian')
    kde.fit(np.array(x)[:, None]) 
    return  kde.score_samples



def stability_measure(data,location,threshold, if_plot = 0): #return the stability measure, input is the data we want to compute and the threshold
    x = [-_[-location] for _ in data]   
    lambda_star = optimize.bisect(robustness_estimator_grad,0, 100000,args=(threshold,x), maxiter=100000000,xtol=0.0000001)
    exp_labmda_x = np.exp([lambda_star*_ for _ in x]) 
    stability_measure = lambda_star * threshold - np.log(np.mean(exp_labmda_x)) 
        
    return {'stability_measure':round(stability_measure,5),'mean':round(np.mean(x),2),'std':round(np.std(x),2),'lambda_star':round(lambda_star,14)}

a_dist_1 = [
{"type": random_half_normal, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}},
{"type": random_exponential, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}}]

a_dist_2 = [
{"type": random_half_normal, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}},
{"type": random_exponential, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}}]

a_dist_3 = [
{"type": random_half_normal, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}},
{"type": random_exponential, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.2)}}]


s_dist_1 = [
{"type": random_exponential, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.8)}}]

s_dist_2 = [
{"type": random_exponential, "kwargs": {"rate": customized_function_generator(func = 'constant',x0 = 0.7)}}]

s_dist_3 = [
{"type": random_exponential, "kwargs": {"rate" :customized_function_generator(func = 'constant',x0 = 0.9)}}]


arrival_distribution = [a_dist_1,a_dist_2,a_dist_3]
service_distribution = [s_dist_1,s_dist_2,s_dist_3]


# note that service has no mixutre
mixture_prob_arrival = [customized_function_generator(func = 'constant',x0 = 0.5),customized_function_generator(func = 'constant',x0 = 0.5)]
mixture_prob_service = [customized_function_generator(func = 'constant',x0 = 1)]

## run simulation 1000 times compare cmu and FIFO, using new code
cost_vector = [1,3,6]
max_age = 1000000
time_len = 100

env = Queueing_Process(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,reward_type='c*age')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DQN(env.observation_space.shape, env.action_space.n,nn_dim = 4).to(device)
model.load_state_dict(torch.load('/user/ym2865/Result/model_file/model_train_3920_2022-01-28_04:01:39.pkl'))
model.eval()


time_len = 100

xlist = list(range(time_len))


def evaluate_policy_under_distribution(arrival_distribution_new, service_distribution_new, mixture_prob_arrival_new, mixture_prob_service_new,plot_text,if_data=0,data=0,if_sin=0,number_of_simulations = 10000,if_save_last = 0):

    os.chdir('/user/ym2865')

    if if_data == 1: # use available data
        c_mu_result = data['c_mu_result']
        FIFO_result = data['FIFO_result']
        DRL_result = data['DRL_result']
    
    else:
        c_mu_result = Parallel(n_jobs= 8)(delayed(map_to_all_discrete_time)(simulate_policy(cost_vector, arrival_distribution_new, service_distribution_new, mixture_prob_arrival_new, mixture_prob_service_new, max_age,'c_mu',time_len ,if_access_all_mu = 0),'cum_reward',xlist,-1) for item in range(number_of_simulations))
        FIFO_result = Parallel(n_jobs= 8)(delayed(map_to_all_discrete_time)(simulate_policy(cost_vector, arrival_distribution_new, service_distribution_new, mixture_prob_arrival_new, mixture_prob_service_new, max_age,'FIFO',time_len),'cum_reward',xlist,-1) for item in range(number_of_simulations))
        DRL_result = Parallel(n_jobs= 8)(delayed(map_to_all_discrete_time)(simulate_policy(cost_vector, arrival_distribution_new, service_distribution_new, mixture_prob_arrival_new, mixture_prob_service_new, max_age,'DRL',time_len,DRL_model = model),'cum_reward',xlist,-1) for item in range(number_of_simulations))


    
    c_mu_result_mean = -np.mean(c_mu_result,axis=0)   
    FIFO_result_mean = -np.mean(FIFO_result,axis=0)   
    DRL_result_mean = -np.mean(DRL_result,axis=0)   
    
    print('c_mu_result_mean',c_mu_result_mean[-1],'DRL_result_mean',DRL_result_mean[-1],'FIFO_result_mean',FIFO_result_mean[-1])
    
    c_mu_result_std = 1.96*np.std(c_mu_result,axis=0) * 1/math.sqrt(number_of_simulations)
    FIFO_result_std = 1.96*np.std(FIFO_result,axis=0)  * 1/math.sqrt(number_of_simulations) 
    DRL_result_std = 1.96*np.std(DRL_result,axis=0) * 1/math.sqrt(number_of_simulations)

    #only save last elements
    if if_save_last == 1:
        c_mu_result = [-1*_[-1] for _ in c_mu_result]
        FIFO_result = [-1*_[-1] for _ in FIFO_result]
        DRL_result = [-1*_[-1] for _ in DRL_result]

    utc_time = datetime.datetime.now()
    str_time = utc_time.strftime("%Y-%m-%d_%H:%M:%S")
    str_time =  '_'+ str(str_time)  
    if if_data == 0: # not use available data
        all_data = {'c_mu_result':c_mu_result,'FIFO_result':FIFO_result,'DRL_result':DRL_result}
        joblib.dump(all_data,'./Result/model_data/'+plot_text+'_data_'+str_time+'.pkl')

    line_width = 7
    fig_size = 47
 
    plt.rcParams["figure.figsize"] = (15,11)
    
    plt.plot(xlist, c_mu_result_mean,'r',label='Gc$-\mu$', linestyle='-',linewidth = line_width) 
    plt.plot(xlist, FIFO_result_mean,'cyan',label='FIFO', linestyle='--',linewidth = line_width) 
    plt.plot(xlist, DRL_result_mean,'orange',label='DQN', linestyle='dotted',linewidth = line_width) 
    
      
    
    plt.fill_between(xlist, c_mu_result_mean-c_mu_result_std, c_mu_result_mean+c_mu_result_std,color='r',alpha=0.15) 
    plt.fill_between(xlist, FIFO_result_mean-FIFO_result_std, FIFO_result_mean+FIFO_result_std,color='cyan',alpha=0.15) 
    plt.fill_between(xlist, DRL_result_mean-DRL_result_std, DRL_result_mean+DRL_result_std,color='orange',alpha=0.15) 
    
    
    
    plt.xlabel('Time', fontsize=fig_size)
    plt.ylabel('Cumulative cost', fontsize=fig_size)
    plt.legend(prop={'size': fig_size}) 
    plt.tick_params(axis='x', labelsize=fig_size-2)
    plt.tick_params(axis='y', labelsize=fig_size-2)
    plt.savefig('./Result/model_data/'+plot_text+'_1.pdf',format='pdf',dpi=1200,bbox_inches='tight')
    plt.show()
    
    if if_sin == 0:
        x = np.arange(0,100,0.1)   # start,stop,step
    else:
        x = np.arange(0,1,0.001)
    traffic_intensity_rate_at_t = []

    for i in range(3):
        arrival_rate_at_t = [arrival_distribution_new[i][0]['kwargs']['rate'](_) for _ in x]
        service_rate_at_t = [service_distribution_new[i][0]['kwargs']['rate'](_) for _ in x]
        temp = [(x/y)for (x,y) in zip(arrival_rate_at_t, service_rate_at_t)]
        traffic_intensity_rate_at_t.append(temp)

    plt.rcParams["figure.figsize"] = (15,11)
    
    plt.plot(x, traffic_intensity_rate_at_t[0], linestyle='-',color='m',label='job type 1',linewidth = line_width) 
    plt.plot(x, traffic_intensity_rate_at_t[1], linestyle='--',color='g',label='job type 2',linewidth = line_width) 
    plt.plot(x, traffic_intensity_rate_at_t[2], linestyle='-.',color='b',label='job type 3',linewidth = line_width) 


    plt.xlabel('Time', fontsize=fig_size)
    plt.ylabel(r'Traffic intensity', fontsize=fig_size)
    plt.legend(prop={'size': fig_size},loc='upper left') 
    plt.tick_params(axis='x', labelsize=fig_size-2)
    plt.tick_params(axis='y', labelsize=fig_size-2)
    
    plt.savefig('./Result/model_data/'+plot_text+'_2.pdf',format='pdf',dpi=1200,bbox_inches='tight')
    plt.show()
    
    print(str_time)

 