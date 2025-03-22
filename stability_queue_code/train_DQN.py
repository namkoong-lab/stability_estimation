#this is the script to generate the trained DQN model model_train_3920_2022-01-28_04/01/39.pkl

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
from collections import deque

import warnings
warnings.filterwarnings("ignore")

from Stability_queue_functions import *


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



def DRL_Q_train(paras, model_id, if_plot = 0, number_of_simulations = 1000):

    ### train_DRL  
    max_age = 100000
   

    env = Queueing_Process(cost_vector, arrival_distribution, service_distribution
                    ,mixture_prob_arrival, mixture_prob_service, max_age,paras['reward_type'])
    
    reset_paras = {'cost_vector':cost_vector,'arrival_distribution':arrival_distribution
                   ,'service_distribution':service_distribution,'mixture_prob_arrival':mixture_prob_arrival
                   ,'mixture_prob_service':mixture_prob_service,'reward_type':paras['reward_type']}

    agent = DQNAgent(env, use_conv=False,learning_rate = paras['learning_rate']
                     , gamma =  paras['gamma'],  nn_dim =  paras['nn_dim']
                     , loss_type = paras['loss_type'], opt_method = paras['opt_method']
                     , weight_decay_val = paras['weight_decay_val'],  momentum_val = paras['momentum_val']
                    )
    trained_DRL_model = mini_batch_train(env, agent, paras['MAX_EPISODES'], paras['MAX_STEPS'], paras['BATCH_SIZE'], reset_paras,eps_val = paras['eps_val'])

    ##save and load the model

    utc_time = datetime.datetime.now()
    str_time = utc_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    model_id = str(model_id) +'_'+ str(str_time) ## add model time, to better keep track of everything
    
    torch.save(trained_DRL_model.state_dict(), './model_file/model_train_'+str(model_id)+'.pkl')
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = trained_DRL_model
 
  
 
    xlist = list(range(time_len))


    c_mu_result = [map_to_all_discrete_time(simulate_policy(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,'c_mu',time_len),'cum_reward',xlist,-1) for item in range(number_of_simulations)]
    FIFO_result = [map_to_all_discrete_time(simulate_policy(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,'FIFO',time_len),'cum_reward',xlist,-1) for item in range(number_of_simulations)]
    DRL_result = [map_to_all_discrete_time(simulate_policy(cost_vector, arrival_distribution, service_distribution, mixture_prob_arrival, mixture_prob_service, max_age,'DRL',time_len,DRL_model = model),'cum_reward',xlist,-1) for item in range(number_of_simulations)]


    c_mu_result_mean = np.mean(c_mu_result,axis=0)   
    FIFO_result_mean = np.mean(FIFO_result,axis=0)   
    DRL_result_mean = np.mean(DRL_result,axis=0)   


    
    fields=['model_id',model_id,'c_mu_result_mean',round(c_mu_result_mean[-1],2),'FIFO_result_mean',round(FIFO_result_mean[-1],2),'DRL_result_mean',round(DRL_result_mean[-1],2),'model_paras',str(paras),'time',str_time]

    if if_plot == 1:
        
        print(fields)
        plt.figure()
        plt.plot(xlist, c_mu_result_mean,'r',label='c_mu') 
        plt.plot(xlist, FIFO_result_mean,'b',label='FIFO')
        plt.plot(xlist, DRL_result_mean,'g',label='DRL')

        plt.title('Compare different policies, test = train')
        plt.xlabel('Time')
        plt.ylabel('Cumulative reward')
        plt.legend() 
        plt.savefig('./plot/Cumulative reward_'+ str(model_id) +'.png')
        plt.show()

        plt.figure()
        plt.plot(range(len(agent.loss_list)), agent.loss_list,'r')
        plt.title('Loss value') 
        plt.savefig('./plot/Loss_'+ str(model_id) +'.png')
        plt.show()




    with open(r'/user/ym2865/result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        

os.chdir('/user/ym2865/Result')

 
paras_dict={'learning_rate':[1/np.power(10,_) for _ in list(range(1,5))]
            ,'opt_method':['Adam','SGD']
            ,'nn_dim':[3,4,10]
            ,'weight_decay_val':[0] + [1/np.power(10,_) for _ in list(range(1,5))]
            ,'momentum_val':[0.9,0]
            ,'gamma':[0.9,0.95],'nn_dim':[4]
            ,'MAX_EPISODES':[40,80]
            ,'MAX_STEPS':[5000,10000,30000]
            ,'eps_val':[0.05]
            ,'BATCH_SIZE':[1,10,100]
            ,'loss_type':['HuberLoss','MSE']
            ,'reward_type':['c*age','cum_diff'] 
           }

 
'''
paras_dict={'learning_rate':[1/np.power(10,_) for _ in list(range(1,5))]
            ,'gamma':[0.9],'nn_dim':[34],'MAX_EPISODES':[10]
             ,'opt_method':['Adam','SGD']
            ,'weight_decay_val':[0] + [1/np.power(10,_) for _ in list(range(1,5))]
            ,'momentum_val':[0.9,0]
            ,'MAX_STEPS':[10],'eps_val':[0.05]
            ,'BATCH_SIZE':[1],'loss_type':['HuberLoss','MSE']
            ,'reward_type':['cum_diff'] 
           }
  
'''

keys, values = zip(*paras_dict.items())
permutations_paras_dict = [dict(zip(keys, v)) for v in product(*values)] 
permutations_paras_dict = dict(zip(range(len(permutations_paras_dict)), permutations_paras_dict))


list_iter_index = list(range(len(permutations_paras_dict)))
random.shuffle(list_iter_index)


#random.seed(100)

#Parallel
results = Parallel(n_jobs=-2)(delayed(DRL_Q_train)(permutations_paras_dict[i],model_id = i, if_plot = 1, number_of_simulations = 1000) for i in list_iter_index)

## test if the code can run
i=0
DRL_Q_train(permutations_paras_dict[i],model_id = i, if_plot = 1, number_of_simulations = 10) 


# try all parameters
for i in list_iter_index:
   DRL_Q_train(permutations_paras_dict[i],model_id = i, if_plot = 1) 
