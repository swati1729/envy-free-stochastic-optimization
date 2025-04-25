# -*- coding: utf-8 -*-
"""
Created on Oct 25 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

Dependency: run RevenueCurveFromData.py before
running this file.

This file contains an implementation of Hazan 
and Levy's bandit convex optimization algorithm. 
Running this file will generate Fig. 5 (center) from 
"Algorithmic Challenges in Ensuring Fairness 
at the Time of Decision." 
"""


import numpy as np
import matplotlib.pyplot as plt
from fanchart import FanChart
import sys 

""" PARAMETERS--EDIT HERE """
T = 3000 # time horizon
runs = 50 # number of runs
std_noise = .4 # standard deviation for mean-0 normal noise 
""" END PARAMETERS """

def BCOHelper(runs, T, std_noise):
    # set random seed and random number generator
    np.random.seed(0)
    rng = np.random.default_rng()

    # load the revenue curves
    file_d = open("d","rb")
    d = np.load(file_d)
    
    # number of products/revenue curves
    num_items = len(d[0,:])
    
    # initialize arrays for algorithm output
    price_paths = np.zeros((runs*num_items,T))
    g = np.zeros((runs*num_items,T))
    nonmonotonicity_0 = np.zeros((runs*num_items,T))
    
    # smoothness calculation
    def smoothness(f, a, b, precision = .001):
        current_smoothness = 0
        for i in range(int(np.floor((b-a)/precision))):
            interval_size = (i+1) * precision 
            for j in range(int(np.floor((b-a)/interval_size))):
                current_smoothness = max(current_smoothness,np.abs(f(a + (j+1)*interval_size) - f(a + j*interval_size) )/interval_size)
        return current_smoothness
    
    # strong convexity calculation
    def strong_convexity(f, a, b, precision = .001):
        current_convexity = np.abs(f(a + precision) - f(a) )/precision
        for i in range(int(np.floor((b-a)/precision))):
            interval_size = (i+1) * precision 
            for j in range(int(np.floor((b-a)/interval_size))):
                current_convexity = min(current_convexity,np.abs(f(a + (j+1)*interval_size) - f(a + j*interval_size) )/interval_size)
        return current_convexity
    
    for item in range(num_items):
        print(f'\nITEM {item+1}/{num_items}')
        
        # min and max decisions
        p_min = 0
        p_max = 1
        
        # golden section search for finding the minimizer
        def gss(f, a, b, tol=1e-5):
            gr = (np.sqrt(5) + 1) / 2 
            
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            while abs(b - a) > tol:
                if f(c) < f(d):
                    b = d
                else:
                    a = c
        
                c = b - (b - a) / gr
                d = a + (b - a) / gr
        
            return (b + a) / 2
    
        # revenue curve for current item
        def f_unscaled(arg):
            ret = 0
            for i in range(len(d[:,0])):
                ret += d[i][item] * arg**(1-i)
            return -ret*arg 
        
        # calculation of the optimum
        optimal_x_unscaled = gss(f_unscaled,p_min,p_max)
        optimal_objective_unscaled = f_unscaled(optimal_x_unscaled)
        worst_objective = max(f_unscaled(p_min), f_unscaled(p_max))
        
        # normalize the revenue curve
        def f(arg):
            return (f_unscaled(arg) - optimal_objective_unscaled)/(worst_objective - optimal_objective_unscaled)
        
        ####################################
        ### HAZAN & LEVY'S BCO ALGORITHM ###
        ####################################

        # self-concordant barrier
        concordant_left = p_min
        concordant_right = p_max 
        def R(x):
            return 10*(-np.log(x-concordant_left) - np.log(-x+concordant_right))
        def ddR(x):
            return 10*(1/((x-concordant_left)**2) + 1/((x-concordant_right)**2)) 
        
        # dimension
        n = 1 
        
        # diameter of price space
        diameter = p_max - p_min    
        
        # difference quotient
        def derivative(f,x,h=0.01):
            return (f(x + h) - f(x - h))/(2*h)
        
        # numerical derivative
        def f_prime(x):
            return derivative(f,x)
        
        # calculation of the smoothness and convexity parameters
        beta = smoothness(f_prime,p_min,p_max)
        alpha = strong_convexity(f_prime,p_min,p_max)
        
        # additional constants for the algorithm
        nu = 2 
        L = max(f(p_min)+.5,f(p_max)+.5)
        eta = np.sqrt(((nu + 2*beta/alpha)*np.log(T))/(2*n*n*L*L*T))
        
        # formatting for plot title
        precision_for_display = 2
        parameters = 'Parameters: T=' + str(T) +  ', $\\alpha=$' + str(round(alpha,precision_for_display)) + ', $\\beta=$' + str(round(beta,precision_for_display)) + ', $\\nu=$' + str(round(nu,precision_for_display)) + ', \n $p_{\min}=$' + str(round(p_min,precision_for_display)) + ', $p_{\max}=$' + str(round(p_max,precision_for_display)) + ', ' + '$\eta$=' + str(round(eta,precision_for_display)) + ', ' + '$\sigma$=' + str(round(std_noise,precision_for_display)) + ', ' + 'runs=' + str(round(runs,precision_for_display))   
        
        # discretization
        precision = .01
        tempp = 1 #0.1
        discretization_size = int(round(diameter/precision,0))
        x_values = np.linspace(p_min + tempp*diameter/discretization_size,p_max - tempp*diameter/discretization_size,discretization_size-2)
        
        # useful array for the algorithm
        eta_R_x = np.linspace(p_min + diameter/discretization_size,p_max - diameter/discretization_size,discretization_size-2)
        for i in range(len(eta_R_x)):
            eta_R_x[i] = (1/eta) * R(x_values[i])
            
        # useful function for update step in the algorithm
        def F_arr(x, g_arr, p_arr):
            return x * np.sum(g_arr) + (alpha/2) * np.sum((x - p_arr)**2)
        vF_arr = np.vectorize(F_arr)
        
        for r in range(runs):
            # generate noise
            noise = np.random.normal(0,std_noise,T)
            
            # progress bar
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*np.ceil(20*(r+1)/runs).astype(int), np.ceil(100*(r+1)/runs)))
            sys.stdout.flush()

            # initial price
            # price_paths[item*runs + r][0] = (p_min + p_max)/2
            price_paths[item*runs + r][0] = rng.uniform(0,1)

            # random integers for gradient estimates
            u_array = rng.integers(2, size=T-1)
            
            
            for t in range(T-1):
                B_t = 1/np.sqrt(ddR(price_paths[item*runs + r][t]) + eta*alpha)
                btscale = 1.2

                # random direction
                u = -1 + 2 * u_array[t]

                if u == 1:
                    B_t = min(B_t, (p_max - price_paths[item*runs + r][t])/btscale)
                else:
                    B_t = min(B_t, (price_paths[item*runs + r][t] - p_min)/btscale)

                # auxiliary point
                y = price_paths[item*runs + r][t] + B_t * u

                # revenue observation at auxiliary point
                revenue = f(y) + noise[t]
                    
                # gradient estimate
                g[item*runs + r][t] = n * (revenue) * (1/B_t) * u
                
                # evaluate F at each price point
                F_array = np.zeros(len(x_values))
                for i in range(len(x_values)):
                    F_array[i] = F_arr(x_values[i], g[item*runs + r,:t+1], price_paths[item*runs + r,:t+1]) + eta_R_x[i]

                    
                # find minimizer of F and update price accordingly
                argmin_F = np.argmin(F_array)
                price_paths[item*runs + r][t+1] = x_values[argmin_F]
                
                # track monotonicity violations
                if price_paths[item*runs + r][t] - price_paths[item*runs + r][t+1] > 0 * (p_max - p_min):
                    nonmonotonicity_0[item*runs + r][t+1] = nonmonotonicity_0[item*runs + r][t] + 1
                else:
                    nonmonotonicity_0[item*runs + r][t+1] = nonmonotonicity_0[item*runs + r][t]
                
                


    
    # density band parameters
    band_sizes = [.2, .5]
    
    # plot non-monotonicity
    fig, ax = plt.subplots() 
    average_nonmonotonicity, banded_low, banded_high = FanChart(nonmonotonicity_0,band_sizes)
    ax.plot(range(T),np.mean(nonmonotonicity_0, axis=0),'-g', label='0%')
    for b in range(len(band_sizes)):
        ax.fill_between(range(T), banded_low[b,:T], banded_high[b,:T], facecolor='green', alpha=.4-b*0.2)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Cumulative non-monotonic jumps', fontsize=10)
    ax.set_title('Hazan & Levy Non-Monotonicity')
    # plt.savefig('fig5-center', dpi=350)
    plt.show()
    

BCOHelper(runs, T, std_noise)

    