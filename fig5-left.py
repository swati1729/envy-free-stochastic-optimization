# -*- coding: utf-8 -*-
"""
Created on May 27 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

Dependency: run RevenueCurveFromData.py before
running this file.

This file contains an implementation of the UCB
algorithm, run on a discretization of the decision
space. Running this file will generate Fig. 5 
(left) from "Algorithmic Challenges in Ensuring 
Fairness at the Time of Decision." 
"""

import numpy as np
import matplotlib.pyplot as plt
from fanchart import FanChart
import sys 
from scipy import stats

""" PARAMETERS--EDIT HERE """
T = 3000 # time horizon
runs = 100 # number of runs
std_noise = .4 # standard deviation for mean-0 normal noise 
""" END PARAMETERS """

# set random seed
np.random.seed(0)

# read in revenue curves
file_d = open("d","rb")
d = np.load(file_d)

# min decision, max decision, and decision space diameter
p_min = 0
p_max = 1
diameter = p_max - p_min

# discretization of decision space
discretization = T**(-1/5)/5
num_prices = int((p_max-p_min)/discretization)
price_space_discretized = np.linspace(p_min,p_max,num_prices)
print(f'Discretized space has {num_prices} prices')

# UCB probability
p = .25

def UCB(n):
    """Finds upper confidence (Hoeffding) bound for n samples"""
    return np.sqrt(-np.log(p)/(2 * n))

# number of retail items
num_items = len(d[0,:])

# initialize array for counts of non-monotonic jumps
nonmonotonicity_0 = np.zeros((runs * num_items,T))


for item in range(num_items):
    print(f'ITEM {item+1}/{num_items}')
    
    # revenue curve for current item
    def f(arg):
        ret = 0
        for i in range(len(d[:,0])):
            ret += d[i][item] * arg**(1-i)
        return ret*arg
    
    # negative revenue curve, if convexity needed
    def neg_f(arg):
        return -f(arg)
    
    # initialize array to store decision path
    price_paths = np.zeros((runs,T))
    
    for r in range(runs):
        num_pulls = np.ones(num_prices) # number of times each price has been chosen
        means = np.ones(num_prices) * 30 # each price's mean is initialized to be large
        
        for t in range(T):  
            # choose a price among those with the largest upper confidence bound
            i = np.random.choice(np.flatnonzero((means + UCB(num_pulls)) == (means + UCB(num_pulls)).max()))

            # update price path
            price_paths[r][t] = price_space_discretized[i]

            # update means and number of pulls
            means[i] = ((num_pulls[i]-1) * means[i] + f(price_paths[r][t]) + np.random.normal(0,std_noise))/num_pulls[i]
            num_pulls[i] += 1
            
            # track monotonicity violations
            if t > 0:
                if price_paths[r][t-1] - price_paths[r][t] > 0 * (p_max - p_min):
                    nonmonotonicity_0[item*runs + r][t] = nonmonotonicity_0[item*runs + r][t-1] + 1
                else:
                    nonmonotonicity_0[item*runs + r][t] = nonmonotonicity_0[item*runs + r][t-1]
                
                
# formatting for plot title
precision_for_display = 2
parameters = 'Parameters: T=' + str(T) + ', $p_{\min}=$' + str(round(p_min,precision_for_display)) + ', $p_{\max}=$' + str(round(p_max,precision_for_display)) + ', ' +  '$\sigma$=' + str(round(std_noise,precision_for_display)) + ', ' + 'runs=' + str(round(runs*num_items,precision_for_display))   

# band sizes for fan chart
band_sizes = [.2, .5]
    
# plot the number of non-monotonic jumps
fig, ax = plt.subplots() 
average_nonmonotonicity, banded_low, banded_high = FanChart(nonmonotonicity_0,band_sizes)
ax.plot(range(T),np.mean(nonmonotonicity_0, axis=0),'-g', label='0%')
for b in range(len(band_sizes)):
    ax.fill_between(range(T), banded_low[b,:T], banded_high[b,:T], facecolor='green', alpha=.4-b*0.2)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Cumulative non-monotonic jumps', fontsize=10)
ax.set_title('$\\bf{Non}$' + '-' +'$\\bf{monotonicity ~of ~DTU}$' + ' \n' + parameters)
# plt.savefig('fig5-left', dpi=350)
plt.show()


