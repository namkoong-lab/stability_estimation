#!/apps/anaconda3/bin/python
# coding: utf-8

# In[2]:


###

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

from Stability_queue_functions import *





# In[3]:


data_0 = joblib.load('./Result/model_data/no distribution shift_500_data__2023-08-31_20:48:17.pkl')

for d in ['c_mu_result','FIFO_result','DRL_result']:
    print (np.mean(data_0[d]))


from random import sample
from collections import defaultdict

def stability_measure(x,threshold): #return the stability measure, input is the data we want to compute and the threshold

    lambda_star = optimize.bisect(robustness_estimator_grad,0, 0.001,args=(threshold,x), maxiter=100000000,xtol=0.0000001)
    exp_labmda_x = np.exp([lambda_star*_ for _ in x]) 
    stability_measure = lambda_star * threshold - np.log(np.mean(exp_labmda_x)) 
        
    return {'stability_measure':round(stability_measure,5),'mean':round(np.mean(x),2),'std':round(np.std(x),2),'lambda_star':round(lambda_star,14)}

thres =  3720.66
n_samples_list = [500,1000,3000,5000,8000,10000,50000,100000]

def partition (list_in, n): 
    #random.Random(123).shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

 
mean_list = defaultdict(list)
std_list = defaultdict(list)

        
for d in ['c_mu_result','FIFO_result','DRL_result']:
    set_seed(123) 
    list_all = partition(data_0[d], 40)
    #print(d,np.mean(list_all, axis = 0))
    for n_sample in n_samples_list: 
        set_seed(123) 
        stability_val_list =  [stability_measure(sample(_,n_sample), thres)['stability_measure']  for _ in list_all]
        mean_list[d].append(np.mean(stability_val_list))
        std_list[d].append(np.std(stability_val_list))
        
    

stability_measure_true = {}
stability_measure_true['c_mu_result'] = 0.15888
stability_measure_true['DRL_result'] = 0.12978
stability_measure_true['FIFO_result'] = 0.0547


policy_list = ['c_mu_result','FIFO_result','DRL_result']
label_list = {'c_mu_result':'Gc-$\mu$','DRL_result':'DQN','FIFO_result':'FIFO'}
color_list = {'c_mu_result':'r','DRL_result':'orange','FIFO_result':'cyan'}


def generate_mse(mean_list,std_list):

    mse_list = defaultdict(list)

    for d in policy_list:
        sig_temp = std_list[d]
        bias_temp = np.subtract(mean_list[d],   [stability_measure_true[d]] * len(sig_temp))

        mse_list[d] = np.power(sig_temp,2) + np.power(bias_temp,2)
    
    return mse_list


mse_list = generate_mse(mean_list,std_list) 

    
plt.figure(figsize=(12,10)) 



linestyle_list = ['bo:','ro-','go-.',':','-'] 
linestyle_list = ['o:','-','-.',':','-'] 
linestyle_list = {'c_mu_result':'-','DRL_result':'dotted','FIFO_result':'--'}

 
line_width = 5 

    
for i, d in enumerate(policy_list):    
    print(d, mse_list[d], stability_measure_true[d])
    plt.plot(n_samples_list,np.log(mse_list[d]),linestyle = linestyle_list[d],
             label =  label_list[d] ,linewidth = line_width, color = color_list[d]) 
 


    
 #'$\sigma$ =' + str(sigma) + ', $y$ = 2, 
    
fig_size = 35

plt.legend() 
plt.tight_layout()   
plt.xlabel('Sample size $n$', fontsize=fig_size)
plt.ylabel('Log of mean squared error', fontsize=fig_size)
plt.legend(prop={'size': fig_size-3},loc='upper right')
 
plt.tick_params(axis='x', labelsize=fig_size)
#plt.xticks([1000,10000,20000,30000,40000],['$10^3$','$10^4$','$2*10^4$','$3*10^4$','$4*10^4$'])
plt.tick_params(axis='y', labelsize=fig_size)

plt.savefig('/user/ym2865/Result/model_data/simulation_stability_queueing.pdf',format='pdf',dpi=1200,bbox_inches='tight')
plt.show()

joblib.dump(mse_list, '/user/ym2865/Result/model_data/simulation_stability_queueing.pkl')


 

