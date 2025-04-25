# -*- coding: utf-8 -*-
"""
Created on Mar  1 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file contains an implementation of the 
one-group Ada-LGD algorithm (Alg. 3) from 
"Algorithmic Challenges in Ensuring Fairness 
at the Time of Decision." This file is used 
as a sub-routine in creating Figure 5 (right).
"""

import numpy as np
import matplotlib.pyplot as plt

def DynamicLaggedGD(T,delta_naught,q,std_noise,n_adjustment,n_min,p_min,p_max,f):
    """Ada-LGD (Algorithm 3)"""

    # set a random seed
    np.random.seed(0)
    
    # set runs to 1 to generate a single sample path
    runs = 1
    
    # Hoeffding probability
    p = T**(-1.6) 
    
    # stopping parameter
    gamma = 1 + 1/np.log(T)     
    
    
    """ ~~~~~~~~~~~~~~~~~~~~~~~ """
    """ ~~~~~~~~Ada-LGD~~~~~~~~ """
    """ ~~~~~~~~~~~~~~~~~~~~~~~ """
    
    def smoothness(f, a, b, precision = .001):
        """Returns the smoothness value of f between a and b"""
        current_smoothness = 0
        for i in range(int(np.floor((b-a)/precision))):
            interval_size = (i+1) * precision 
            for j in range(int(np.floor((b-a)/interval_size))):
                current_smoothness = max(current_smoothness,np.abs(f(a + (j+1)*interval_size) - f(a + j*interval_size) )/interval_size)
        return current_smoothness
    
    def strong_convexity(f, a, b, precision = .001):
        """Returns the strong convexity value of f between a and b"""
        current_convexity = np.abs(f(a + precision) - f(a) )/precision
        for i in range(int(np.floor((b-a)/precision))):
            interval_size = (i+1) * precision 
            for j in range(int(np.floor((b-a)/interval_size))):
                current_convexity = min(current_convexity,np.abs(f(a + (j+1)*interval_size) - f(a + j*interval_size) )/interval_size)
        return current_convexity
    
    def gss(f, a, b, tol=1e-5):
        """Golden Section Search: optimizes f between a and b"""
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
    
    def derivative(f,x,h=0.01):
        return (f(x + h) - f(x - h))/(2*h)
    
    def f_prime(x):
        return derivative(f,x)
    
    # calculation of the smoothness and convexity parameters
    beta = smoothness(f_prime,p_min,p_max)
    alpha = strong_convexity(f_prime,p_min,p_max)
    
    # calculation of the optimum
    optimal_x = gss(f,p_min,p_max)
    optimal_objective = f(optimal_x)
    
    # function value separation required given price gap x
    def epsilon(x):
        return alpha * x * x / 4
    
    xi = 1-q
    def delta(i):
        """Returns the ith lag size"""
        return (q**i) * delta_naught
    
    # number of samples required for a given price gap
    def n(gap):
        return max(int(np.ceil(2*np.log(1/p)/(n_adjustment*epsilon(gap)**2))),n_min)
    
    # gradient calculation given two points
    def gradient(left,right):
        gap = right-left
        N = n(gap)
        samples_left = np.zeros(N)
        samples_right = np.zeros(N) 
        for i in range(N): 
            samples_left[i] = f(left) + np.random.normal(0,std_noise)
            samples_right[i] = f(right) + np.random.normal(0,std_noise)
        return (np.mean(samples_right) - np.mean(samples_left) + epsilon(gap))/gap, samples_left, samples_right
    
    # initialize arrays/variables to store simulation data
    T_condensed = T
    price_paths = np.zeros((runs,T_condensed))
    lagged_price_paths = np.zeros((runs,T_condensed))
    complete_price_paths = np.zeros((runs,T_condensed))
    uncondensed_price_paths = np.zeros((runs,T))
    eta = (1/beta) * np.ones(T_condensed)
    x_nonlagged = np.zeros(T)
    first_lagged_point = np.zeros(T)
    
    # formatting for plot title
    precision_for_display = 2
    parameters = 'Parameters: T=' + str(T) + ', $n_{adj}=$' + '{:.1E}'.format(n_adjustment) + ', $n_{\min}=$' + str(n_min) + ', $\delta_0=$' + str(round(delta_naught,precision_for_display)) + ', ' +'\n' +'$q=$' + str(round(q,precision_for_display))  +', $\\alpha=$' + '{number:.{digits}f}'.format(number=alpha, digits=precision_for_display) + ', $\\beta=$' + str(round(beta,precision_for_display)) + ', $p_{\min}=$' + str(round(p_min,precision_for_display)) + ', $p_{\max}=$' + str(round(p_max,precision_for_display)) + ', ' + '$\sigma$=' + str(round(std_noise,precision_for_display)) #+ ', ' + '$\eta$=' + str(round(eta[0],precision_for_display)) #+ ', gradient-threshold=' + str(round(beta*(2+(.5+ diameter)*beta)*delta/threshold_adjustment,precision_for_display))  
    
    for r in range(runs):
        # variables for tracking number of samples of various types
        total_samples_so_far = 0
        nonlagged_index = 0 
        lagged_index = 0 
        complete_index = 0
        uncondensed_index = 0 
        first_lagged_index = 0
        
        # number of lag size decreases
        phase = 0 

        # initial non-lagged/lagged point
        price_paths[r][nonlagged_index] = p_min + delta(phase)
        first_lagged_point[first_lagged_index] = price_paths[r][nonlagged_index] - delta(phase)

        # gradient estimate initialized to 0
        g = 0

        while total_samples_so_far < T:
            g = 0
            phase = phase - 1

            # while the gradient is too small compared to the current lag size...
            while -eta[nonlagged_index] * g < (2+gamma) * delta(phase) and total_samples_so_far < T:
                phase = phase + 1

                # if there is enough remaining time for the required samples...
                if 2*n(xi * delta(phase)) < T - total_samples_so_far:
                    # estimate the gradient
                    g,samples_left,samples_right = gradient(price_paths[r][nonlagged_index] - delta(phase),price_paths[r][nonlagged_index] - delta(phase+1))

                    # update trackers
                    total_samples_so_far = total_samples_so_far + 2*n(xi*delta(phase))
                    lagged_price_paths[r][lagged_index] = price_paths[r][nonlagged_index] - delta(phase)
                    lagged_price_paths[r][lagged_index+1] = price_paths[r][nonlagged_index] - delta(phase+1)
                    lagged_index = lagged_index + 2
                    complete_price_paths[r][complete_index] = price_paths[r][nonlagged_index] - delta(phase)
                    complete_price_paths[r][complete_index+1] = price_paths[r][nonlagged_index] - delta(phase+1)
                    complete_index = complete_index + 2
                    uncondensed_price_paths[r][uncondensed_index:uncondensed_index+n(xi*delta(phase))] = (price_paths[r][nonlagged_index] - delta(phase)) * np.ones(n(xi * delta(phase)))
                    uncondensed_price_paths[r][uncondensed_index+n(xi*delta(phase)):uncondensed_index+2*n(xi*delta(phase))] = (price_paths[r][nonlagged_index] - delta(phase+1)) * np.ones(n(xi * delta(phase)))
                    uncondensed_index = uncondensed_index + 2*n(xi*delta(phase))
                
                # if there is only enough time to fully sample at one more lagged point...
                elif n(xi * delta(phase)) < T - total_samples_so_far:
                    # sample at that lagged point, move to the next, and update trackers
                    lagged_price_paths[r][lagged_index] = price_paths[r][nonlagged_index] - delta(phase)
                    lagged_price_paths[r][lagged_index+1] = price_paths[r][nonlagged_index] - delta(phase+1)
                    lagged_index = lagged_index + 2
                    complete_price_paths[r][complete_index] = price_paths[r][nonlagged_index] - delta(phase)
                    complete_price_paths[r][complete_index+1] = price_paths[r][nonlagged_index] - delta(phase+1)
                    complete_index = complete_index + 2
                    uncondensed_price_paths[r][uncondensed_index:uncondensed_index+n(xi*delta(phase))] = (price_paths[r][nonlagged_index] - delta(phase)) * np.ones(n(xi * delta(phase)))
                    uncondensed_price_paths[r][uncondensed_index+n(xi*delta(phase)):T] = (price_paths[r][nonlagged_index] - delta(phase+1)) * np.ones(T - (uncondensed_index+n(xi*delta(phase))))
                    uncondensed_index = T
                    total_samples_so_far = T

                # if there isn't enough time to fully sample at the next lagged point...
                else: 
                    # move to that lagged point and update trackers
                    lagged_price_paths[r][lagged_index] = price_paths[r][nonlagged_index] - delta(phase)
                    lagged_index = lagged_index + 1
                    complete_price_paths[r][complete_index] = price_paths[r][nonlagged_index] - delta(phase)
                    complete_index = complete_index + 1
                    uncondensed_price_paths[r][uncondensed_index:T] = (price_paths[r][nonlagged_index] - delta(phase)) * np.ones(T - uncondensed_index)
                    uncondensed_index = T
                    total_samples_so_far = T
            

            # update final points and trackers when time horizon is reached
            if uncondensed_index < T:
                complete_price_paths[r][complete_index] = price_paths[r][nonlagged_index]
                complete_index = complete_index + 1
                x_nonlagged[nonlagged_index] = total_samples_so_far

            
            if n(delta(phase)) < T - total_samples_so_far:
                mean_lagged = np.mean(samples_left[:n(delta(phase))])
                samples_right = np.zeros(n(delta(phase))) 
                for i in range(len(samples_right)):
                    samples_right[i] = f(price_paths[r][nonlagged_index]) + np.random.normal(0,std_noise)
                g = (np.mean(samples_right) - mean_lagged + epsilon(delta(phase)))/delta(phase)
                
                total_samples_so_far = total_samples_so_far + n(delta(phase))
                if - eta[nonlagged_index] * g - 2*delta(phase) >= 0:
                    price_paths[r][nonlagged_index+1] = price_paths[r][nonlagged_index] - eta[nonlagged_index] * g - delta(phase)
                    first_lagged_point[first_lagged_index+1] = total_samples_so_far
                    first_lagged_index = first_lagged_index + 1
                    uncondensed_price_paths[r][uncondensed_index:uncondensed_index+n(delta(phase))] = (price_paths[r][nonlagged_index] ) * np.ones(n(delta(phase)))
                    uncondensed_index = uncondensed_index + n(delta(phase))
                    nonlagged_index = nonlagged_index + 1
                else:
                    uncondensed_price_paths[r][uncondensed_index:T] = (price_paths[r][nonlagged_index] ) * np.ones(T-uncondensed_index)
                    uncondensed_index = T
                    total_samples_so_far = T
                        
            else:
                uncondensed_price_paths[r][uncondensed_index:T] = (price_paths[r][nonlagged_index] ) * np.ones(T-uncondensed_index)
                uncondensed_index = T
                nonlagged_index = nonlagged_index + 1
                total_samples_so_far = T
        
    return uncondensed_price_paths,price_paths,optimal_x,x_nonlagged,first_lagged_point,parameters,f
           
