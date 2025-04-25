# -*- coding: utf-8 -*-
"""
Created on May 12 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file creates Figure 5 (right) in "Algorithmic
Challenges in Ensuring Fairness at the Time 
of Decision", depicting a trajectory (one
sample path) of the one-group Ada-LGD
algorithm (Alg. 3).
"""

import numpy as np
import matplotlib.pyplot as plt
from AdaLGD import DynamicLaggedGD

# read in revenue curves
file_d = open("d","rb")
d = np.load(file_d)

# generate a sample path for item 0
item = 0
print('ITEM ',str(item))

# min and max prices
p_min = 0
p_max = 1

# initial lag size
delta_naught = .1

# negative revenue curve
def f_linear(arg):
    ret = 0
    for i in range(len(d)):
        ret += d[i][item] * arg**(1-i)
    return -ret*arg

# negative revenue curve, flipped horizontally
def f_flipped(arg):
    return f_linear(p_min + p_max - arg)
    
# time horizon
T = 300 

# contraction in lag size
q = 3/4 

# for mean-0 normal noise 
std_noise = .4 

# adjustments to reduce/control the number of samples
n_adjustment = 1000000
n_min = 10

# run Ada-LGD
uncondensed_price_paths,price_paths,optimal_x,x_nonlagged,first_lagged_point,parameters,f = DynamicLaggedGD(T,delta_naught,q,std_noise,n_adjustment,n_min,p_min,p_max,f_flipped)

# find optimum for flipped function
optimal_x = p_min + p_max - optimal_x

# find flipped price paths
price_paths = p_min + p_max - price_paths
uncondensed_price_paths = p_min + p_max - uncondensed_price_paths
      
# plot price paths
fig, ax = plt.subplots() 
ax.plot(range(len(uncondensed_price_paths[0,:])),np.mean(uncondensed_price_paths, axis=0),'-g', label='Price path')
ax.plot(x_nonlagged,np.mean(price_paths[:,:len(x_nonlagged)], axis=0),'r^', label='Non-lagged prices')
ax.plot(range(T),optimal_x*np.ones(T),'k--', label='Optimal price')
for i in range(len(price_paths[0,:])):
    if i < len(x_nonlagged):
        if i==0:
            ax.plot(np.linspace(first_lagged_point[i] ,x_nonlagged[0],20), price_paths[0][i]*np.ones(20),'r--',linewidth=.8)
        else:
            ax.plot(np.linspace(first_lagged_point[i],x_nonlagged[i],20), price_paths[0][i]*np.ones(20),'r--',linewidth=.8)
    elif i>0:
        ax.plot(np.linspace(first_lagged_point[i],T,20), price_paths[0][i]*np.ones(20),'r--',linewidth=.8)
    else:
        ax.plot(np.linspace(0,T,20), price_paths[0][i]*np.ones(20),'r--',linewidth=.8)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Price', fontsize=10)
ax.set_title('$\\bf{Price ~path~ plotted ~over~ time}$' + ' \n' + parameters)
plt.legend(loc='upper right')
# ax.set_ylim([2.5,7.5])
plt.savefig("fig5-right.png", dpi=350)
plt.show()


