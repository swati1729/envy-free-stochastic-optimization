# -*- coding: utf-8 -*-
"""
Created on Jan 26 2022
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file creates Figure 6 in "Algorithmic
Challenges in Ensuring Fairness at the Time 
of Decision", depicting a trajectory (one
sample path) of the two-group SCAda-LGD
algorithm (Alg. 4).
"""

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from MyPriorityQueue import MyPriorityQueue

""" PARAMETERS """
gradient_threshold = 10 # multiplicative adjustment to gradients
jump_adjustment = .9 # multiplicative adjustment to jump sizes
runs = 1 # number of runs (1 to generate a single sample path)
T = 10000 # time horizon
n_min = 200 # minimum sample size
n_adjustment = 750_000 # multiplicative adjustment to sample size
delta_naught = .07 # initial lag size
std_noise = .4 # standard deviation for mean-0 normal noise 
""" END PARAMETERS """

def DynamicLaggedGD_2G(T,delta_naught,q,std_noise,n_adjustment,n_min,beta_adjustment,p_min,p_max,f_0,f_1,slack,runs,gradient_threshold,optimal_x_constrained,jump_adjustment):

    
    # Hoeffding probability
    p = T**(-1.6) 

    E_max = np.sqrt(8 * std_noise**2/3)
    
    # stopping parameter
    gamma = 1 + 1/np.log(T) 

    # number of groups    
    num_groups = 2
    
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
    
    def gss(f, a, b, tol=1e-5):
        """Golden Section Search: minimizes a unimodal function"""
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
        
    # difference quotient
    def derivative(f,x,h=0.01):
        return (f(x + h) - f(x - h))/(2*h)
    
    # numerical derivative
    def f_prime_0(x):
        return derivative(f_0,x)
    def f_prime_1(x):
        return derivative(f_1,x)
    
    # array of the two group-specific objective functions
    f = np.array([f_0,f_1])
    
    # calculation of the smoothness and convexity parameters
    beta0 = beta_adjustment * smoothness(f_prime_0,p_min,p_max)
    beta1 = beta_adjustment * smoothness(f_prime_1,p_min,p_max)
    beta = max(beta0,beta1)
    alpha0 = strong_convexity(f_prime_0,p_min,p_max)
    alpha1 = strong_convexity(f_prime_1,p_min,p_max)
    alpha = min(alpha0,alpha1)
    
    # calculation of the optimum
    optimal_x = np.zeros(2)
    optimal_x[0] = gss(f_0,p_min,p_max)
    optimal_x[1] = gss(f_1,p_min,p_max)
    optimal_objective = np.zeros(2)
    optimal_objective[0] = f_0(optimal_x[0])
    optimal_objective[1] = f_1(optimal_x[1])    
    
    # function value separation required given price gap x
    def epsilon(x):
        return alpha * x * x / 4
    
    xi = 1-q
    def delta(i):
        """Returns the ith lag size"""
        return (q**i) * delta_naught
    
    # number of samples required for a given price gap
    def n(gap):
        return min(int(np.log(T)/(n_adjustment * gap**4)) + n_min, T-num_samples_so_far)

    
    num_samples_so_far = 0
    
    def observed_mean(arg,delta,index,f,noise):
        """Sample at arg the required number of times.
        Return the mean and the new number of samples."""
        N = n(delta)
        samples = np.zeros(N)
        for i in range(N):
            samples[i] = f(arg) + noise[index+i]
        return np.mean(samples),index+N
    
    def observed_mean_combined(arg,delta,index,f,noise):
        """Sample at arg the required number of times.
        Return the mean and the new number of samples."""
        N = n(delta)
        samples = np.zeros(N)
        for i in range(N):
            samples[i] = f(arg) + noise[0,index+i] + noise[1,index+i]
        return np.mean(samples),index+N
    
    def grad(left,right,gap):
        """Finds the adjusted secant given the mean at the
        left endpoint and the mean at the right endpoint."""
        eps_adjustment = 10000
        return (right - left + epsilon(gap)/eps_adjustment)/gap
    
    T_condensed = T
    
    # initialize arrays for algorithm output
    price_paths = np.zeros((runs,2,T_condensed)) # runs x group x T 
    price_paths_types = np.zeros((runs,2,T_condensed)) # {0:first-lagged, 1:lagged, 2:non-lagged, 3:first-feas, 4:second-feas}
  
    # formatting for plot title    
    precision_for_display = 2
    parameters = 'Parameters: T=' + str(T) + ', $n_{adj}=$' + '{:.1E}'.format(n_adjustment) + ', $n_{\min}=$' + str(n_min) + ', $\delta_0=$' + str(round(delta_naught,precision_for_display)) + ', ' +'\n' +'$q=$' + str(round(q,precision_for_display))  +', $\\alpha=$' + '{number:.{digits}f}'.format(number=alpha, digits=precision_for_display) + ', $\\beta=$' + str(round(beta,precision_for_display)) + ', $p_{\min}=$' + str(round(p_min,precision_for_display)) + ', $p_{\max}=$' + str(round(p_max,precision_for_display)) + ', ' + '$\sigma$=' + str(round(std_noise,precision_for_display)) #+ ', ' + '$\eta$=' + str(round(eta[0],precision_for_display)) #+ ', gradient-threshold=' + str(round(beta*(2+(.5+ diameter)*beta)*delta/threshold_adjustment,precision_for_display))  
    
    
    """
    CODE FOR QUEUE KEYS--TRACKS TYPE OF EACH POINT
    
    <10: FIRST OF THE LAGGED POINTS
    <20: SECOND OF THE LAGGED POINTS
    <30: NON-LAGGED
    <40: FIRST OF THE FEASIBLE POINTS
    <50: SECOND OF THE FEASIBLE POINTS
    <60: COORDINATE PHASE WAITING
    <70: COMBINED PHASE FIRST OF THE LAGGED POINTS
    <80: COMBINED PHASE SECOND OF THE LAGGED POINTS
    <90: COMBINED PHASE NON-LAGGED
    
    """
    num_combined = 0
    D = {
        0:'First of lags', 
        1:'Second of lags', 
        2:'Non-lag', 
        3:'First feasible', 
        4:'Second feasible', 
        5:'Coordinate waiting', 
        6:'Combined first of lags', 
        7:'Combined second of lags', 
        8:'Combined non-lag'
        }
    for r in range(runs):
        # initialize lists of sampled points for each group
        sampled_x = []
        sampled_y = []
        
        # find combined smoothness and strong convexity values
        beta = max(beta0,beta1)
        alpha = min(alpha0,alpha1)

        # phase for each group
        phase = np.zeros(2)

        # initialize arrays for the sample means at the most
        # recent two points
        recent_means = np.zeros((2,2))
        recent_means_feasibility = np.zeros((2,2))

        # set random seed
        np.random.seed(8)
        
        # generate normally distributed noise
        noise = np.zeros((2,T))
        noise[0,:] = np.random.normal(0,std_noise,T)
        noise[1,:] = np.random.normal(0,std_noise,T)
            
        # create priority queues for points to sample
        queue_0 = MyPriorityQueue()
        queue_1 = MyPriorityQueue()
        queue = np.array([queue_0, queue_1], dtype=MyPriorityQueue)
        
        # insert first prices for both groups
        for group in range(num_groups):
            queue[group].insert({xi*delta(phase[group]): p_min})
            queue[group].insert({xi*(q**2)*delta(phase[group]) + 10: p_min + delta(phase[group]) - q*delta(phase[group])})
            queue[group].insert({xi*q*delta(phase[group]) + 20: p_min + delta(phase[group])})

        # current pair of prices
        current = np.zeros(2)
        current[0] = p_min
        current[1] = p_min
        
        # group to optimize over
        i = 0
        
        num_samples_so_far = 0
        done_optimizing = np.array([0, 0], dtype=bool) # whether each group has reached a low gradient
        
        # find the current maximum price for each group, given slacks
        max_price = np.zeros(2) 
        max_price[0] = current[1] + slack[0,1]
        max_price[1] = current[0] + slack[1,0]
        
        # current gradient for both groups
        g = np.zeros(2)
        
        # whether or not we have entered the combined phase
        combined = False
        
        num_points_queried = 0
        
        ## COORDINATE DESCENT PHASE ##
        while_1_count = 0
        while num_samples_so_far < T and not combined:
            
            j = 1-i # the other group's index
            
            # max price for current group
            max_price[i] = current[j] + slack[i,j]
            
            lag_dropped_below = False
            switch_groups = False
            
            while_1_count += 1
            
            # if we are done optimizing the current group, edit the
            # remaining portion of the price path accordingly
            if done_optimizing[i]:
                for group in range(num_groups):
                    price_paths[r,group,num_samples_so_far:T] = current[group]*np.ones(T-num_samples_so_far)
                    price_paths_types[r,group,num_samples_so_far:T] = 9*np.ones(T-num_samples_so_far)
                num_samples_so_far = T

            # add feasibility iterates to the queue
            if phase[j] > phase[i]:
                for group in range(num_groups):
                    queue[group].insert({xi*delta(phase[j]) + 30:max_price[i] - delta(phase[j])})
                    queue[group].insert({xi*delta(phase[j]) + 40:max_price[i] - q*delta(phase[j])})
            
            while_2_count = 0
                
            # run the main portion of 2G-LGD.

            # while the current group has not reached its optimum, we have not entered the combined
            # phase, and we have not reached the time horizon...
            while queue[i].hasFeasiblePoints(max_price[i]) and not done_optimizing[i] and not switch_groups and not combined and num_samples_so_far < T:
                while_2_count += 1

                current_temp = current[i]

                # get the next point to sample for group i
                lag, current[i] = queue[i].delete()
                
                # next points to sample for both groups
                sampled_x += [current[0]]
                sampled_y += [current[1]]

                # if non-monotonic jump detected, print troubleshooting information...
                if current_temp > current[i]:
                    print('Uh-oh! Non-monotonic jump')
                    print('Group: {}'.format(i))
                    print('NumSamp: {0} out of {1}'.format(num_samples_so_far,T))
                    print('Num points queries: {}'.format(num_points_queried))
                    print('Phases: [{0}, {1}]'.format(phase[0],phase[1]))
                    print('Done optimizing G0: {}'.format(done_optimizing[0]))
                    print('Done optimizing G1: {}'.format(done_optimizing[1]))
                    for point_type in range(9):
                        if lag - point_type*10 < 8 and lag - point_type*10 >= 0:
                            type_of_point = point_type        
                    print(D[type_of_point])
                
                # if the next point is the first of the lagged points of this phase...
                if lag < 8: 
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    price_paths_types[r,i,num_samples_so_far:num_samples_so_far + n(lag)] = 0*np.ones(n(lag))
                    for group in range(num_groups):
                        price_paths[r,group,num_samples_so_far:num_samples_so_far + n(lag)] = current[group]*np.ones(n(lag))
                    
                    # sample and record mean of observations
                    recent_means[i,0],num_samples_so_far = observed_mean(current[i],lag,num_samples_so_far,f[i],noise[i,:])

                # if the next point is lagged, but not the first lagged point...
                elif lag < 18: 
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    price_paths_types[r,i,num_samples_so_far:num_samples_so_far + n(lag-10)] = 1*np.ones(n(lag-10))
                    for group in range(num_groups):
                        price_paths[r,group,num_samples_so_far:num_samples_so_far + n(lag-10)] = current[group]*np.ones(n(lag-10))
                    
                    # sample and record mean of observations
                    recent_means[i,1],num_samples_so_far = observed_mean(current[i],lag-10,num_samples_so_far,f[i],noise[i,:])

                    # estimate the gradient
                    g[i] = grad(recent_means[i,0], recent_means[i,1], xi*delta(phase[i]))
                    
                    # check if the small gradient condition is met
                    if -g[i] < T**(-1/4)/gradient_threshold:
                        done_optimizing[i] = True
                        delta_temp = delta(phase[i]) 
                        while delta(phase[i]) > delta_temp * xi:
                            phase[i] = phase[i] + 1

                    # if gradient is large enough compared to lag size...
                    if -(1/beta) * g[i] < (1+gamma)*delta(phase[i]):
                        recent_means[i,0] = recent_means[i,1]
                        phase[i] = phase[i] + 1 # increase the phase

                        # add next point to sample
                        queue[i].insert({xi * delta(phase[i]) + 10:current[i] + xi * delta(phase[i])})

                        # track which group has smaller lag size
                        if phase[i] > phase[j]:
                            lag_dropped_below = True

                # if the next point is non-lagged...      
                elif lag < 28: 
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    price_paths_types[r,i,num_samples_so_far:num_samples_so_far + n(delta(phase[i]))] = 2*np.ones(n(delta(phase[i])))
                    for group in range(num_groups):
                        price_paths[r,group,num_samples_so_far:num_samples_so_far + n(delta(phase[i]))] = current[group]*np.ones(n(delta(phase[i])))
                    
                    # sample and record mean of observations
                    recent_means[i,1],num_samples_so_far = observed_mean(current[i],delta(phase[i]),num_samples_so_far,f[i],noise[i,:])
                    
                    # estimate the gradient
                    g[i] = grad(recent_means[i,0], recent_means[i,1], delta(phase[i]))

                    # check if the small gradient condition is met
                    if -g[i] < T**(-1/4)/gradient_threshold:
                        done_optimizing[i] = True

                    # if gradient is large enough compared to the lag size...
                    if - (1/beta) * g[i] * jump_adjustment > 2 * delta(phase[i]):
                        queue[i].insert({delta(phase[i]) + 20:current[i] - (1/beta) * g[i] * jump_adjustment}) # add next non-lagged iterate to queue
                        queue[i].insert({xi * delta(phase[i]):current[i] - (1/beta) * g[i] * jump_adjustment - delta(phase[i])}) # add next two lagged iterates to queue
                        queue[i].insert({xi * q * delta(phase[i]) + 10:current[i] - (1/beta) * g[i] * jump_adjustment - q * delta(phase[i])}) 
                    else:
                        done_optimizing[i] = True
                    
                    # if the current group's lag size has become smaller
                    # than the other group's, switch groups
                    if lag_dropped_below:
                        switch_groups = True

                # if next point is a feasibilitiy iterate...
                elif lag < 38:  
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    price_paths_types[r,i,num_samples_so_far:num_samples_so_far + n(lag-30)] = 3*np.ones(n(lag-30))
                    for group in range(num_groups):
                        price_paths[r,group,num_samples_so_far:num_samples_so_far + n(lag-30)] = current[group]*np.ones(n(lag-30))
                    recent_means_feasibility[i,0],num_samples_so_far = observed_mean(current[i],lag-30,num_samples_so_far,f[i],noise[i,:])
                elif lag < 48: # iterate is second feasibility 
                    num_points_queried += 1
                    # num_samples_temp = n(lag-40)

                    # track the next price and the type of the next price
                    price_paths_types[r,i,num_samples_so_far:num_samples_so_far + n(lag-40)] = 4*np.ones(n(lag-40))
                    for group in range(num_groups):
                        price_paths[r,group,num_samples_so_far:num_samples_so_far + n(lag-40)] = current[group]*np.ones(n(lag-40))
                    
                    # sample and record mean of observations
                    recent_means_feasibility[i,1],num_samples_so_far = observed_mean(current[i],lag-40,num_samples_so_far,f[i],noise[i,:])
                    
                    # estimate the gradient
                    g[i] = grad(recent_means_feasibility[i,0], recent_means_feasibility[i,1], lag-40)

                    # if the gradient is large enough...
                    if -(1/beta) * g[i] >= ( ((2+gamma) * beta)/(q * alpha) + 1) * delta(phase[j]):
                        group_at_facet = i
                        combined = True # enter the combined phase
                    
            if not done_optimizing[i] and not switch_groups and not combined:
                current[i] = current[j] + slack[i,j] # move to the facet
            
            i = j
            switch_groups = False
            lag_dropped_below = False
        

        end_combined = False
        ## COMBINED DESCENT PHASE ##
        if combined:
            num_combined = num_combined + 1 # number of combined groups

            i = group_at_facet
            j = 1-i

            # queue for combined points to sample
            queue_combined = MyPriorityQueue()

            # means of the previous two points
            recent_means_combined = np.zeros(2)

            # add the next point to sample to the queue
            queue_combined.insert({delta(phase[j]) + 80:current[j]})

            # get next point to sample from queue
            lag, current = queue_combined.delete()

            # combined gradient
            g = g[0] + g[1]

            # combined objective function
            def h(x):
                return f[i](x + slack[i,j]) + f[j](x)
            
            # combined smoothness/strong convexity values
            beta = 2*beta 
            alpha = 2*alpha
            
            overshoot = min(0,- (1/beta) * g - delta(phase[j]))
            
            # add points to sample to queue
            queue_combined.insert({delta(phase[j]) + 80:current - (1/beta) * g - overshoot})
            queue_combined.insert({xi * delta(phase[j]) + 60:current - (1/beta) * g - delta(phase[j]) - overshoot})
            queue_combined.insert({q * xi * delta(phase[j]) + 70:current - (1/beta) * g - q * delta(phase[j]) - overshoot})
            
            
            while_3_count = 0
            while num_samples_so_far < T and (not end_combined) and (not queue_combined.isEmpty()):
                while_3_count += 1

                # get next point to sample
                current_temp = current
                lag, current = queue_combined.delete()

                # if non-monotonicity detected, print troubleshooting information
                if current_temp > current:
                    print('\nUh-oh! Non-monotonic jump')
                    print('Combined')
                    print('NumSamp: {0} out of {1}'.format(num_samples_so_far,T))
                    print('Num points queries: {}'.format(num_points_queried))
                    print('Phase: {}'.format(phase[j]))
                    for point_type in range(9):
                        if lag - point_type*10 < 8 and lag - point_type*10 >= 0:
                            type_of_point = point_type        
                    print(D[type_of_point])

                # if next point is the first lagged point of current phase...
                if lag < 68: 
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    for group in range(num_groups):
                        price_paths_types[r,group,num_samples_so_far:num_samples_so_far + n(lag-60)] = 6*np.ones(n(lag-60))
                    price_paths[r,j,num_samples_so_far:num_samples_so_far + n(lag-60)] = current*np.ones(n(lag-60))
                    price_paths[r,i,num_samples_so_far:num_samples_so_far + n(lag-60)] = (current+slack[i,j])*np.ones(n(lag-60))
                    
                    # sample and record mean of observations
                    recent_means_combined[0],num_samples_so_far = observed_mean_combined(current,lag-60,num_samples_so_far,h,noise[:,:])
                
                # if next point is lagged but not the first lagged point...
                elif lag < 78: # iterate is (not first) of lags
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    for group in range(num_groups):
                        price_paths_types[r,group,num_samples_so_far:num_samples_so_far + n(lag-70)] = 7*np.ones(n(lag-70))
                    price_paths[r,j,num_samples_so_far:num_samples_so_far + n(lag-70)] = current*np.ones(n(lag-70))
                    price_paths[r,i,num_samples_so_far:num_samples_so_far + n(lag-70)] = (current+slack[i,j])*np.ones(n(lag-70))
                    
                    # sample and record mean of observations
                    recent_means_combined[1],num_samples_so_far = observed_mean_combined(current,lag-70,num_samples_so_far,h,noise[:,:])
                    
                    # estimate gradient
                    g = grad(recent_means_combined[0], recent_means_combined[1], xi*delta(phase[j]))
                    
                    # if we overshot the constrained optimum...
                    if g > 0:
                        end_combined = True

                    # if the gradient is too small compared to the lag size...
                    if -(1/beta) * g < (1+gamma)*delta(phase[j]):
                        recent_means_combined[0] = recent_means_combined[1]
                        phase[j] = phase[j] + 1 # decrease the lag size

                        # add next lagged point to queue
                        queue_combined.insert({xi * delta(phase[j]) + 70:current + xi * delta(phase[j])})
                
                # if next point is non-lagged...
                elif lag < 88: 
                    num_points_queried += 1

                    # track the next price and the type of the next price
                    for group in range(num_groups):
                        price_paths_types[r,group,num_samples_so_far:num_samples_so_far + n(delta(phase[j]))] = 8*np.ones(n(delta(phase[j])))
                    price_paths[r,j,num_samples_so_far:num_samples_so_far + n(delta(phase[j]))] = current*np.ones(n(delta(phase[j])))
                    price_paths[r,i,num_samples_so_far:num_samples_so_far + n(delta(phase[j]))] = (current+slack[i,j])*np.ones(n(delta(phase[j])))
                    
                    # sample and record mean of observations
                    recent_means_combined[1],num_samples_so_far = observed_mean_combined(current,delta(phase[j]),num_samples_so_far,h,noise[:,:])
                    
                    # estimate gradient
                    g = grad(recent_means_combined[0], recent_means_combined[1], delta(phase[j]))

                    # if we overshot the constrained optimum...
                    if g > 0:
                        end_combined = True
                    
                    # if gradient is large enough...
                    if -(1/beta) * g > (1+gamma) * delta(phase[j]):
                        queue_combined.insert({delta(phase[j]) + 80:current - (1/beta) * g}) # add next non-lagged iterate to queue
                        queue_combined.insert({xi * delta(phase[j]) + 60:current - (1/beta) * g - delta(phase[j])}) # add next two lagged iterates to queue
                        queue_combined.insert({xi * q * delta(phase[j]) + 70:current - (1/beta) * g - q * delta(phase[j])})
                    else:
                        end_combined = True
                        
            if num_samples_so_far < T and combined:
                price_paths[r,j,num_samples_so_far:T] = current*np.ones(T-num_samples_so_far)
                price_paths[r,i,num_samples_so_far:T] = (current+slack[i,j])*np.ones(T-num_samples_so_far)


        # make plot    
        fig, ax = plt.subplots()
        x1 = np.linspace(slack[0,1],1,50)
        y1 = np.linspace(0,1-slack[0,1],50)
        ax.plot(x1, y1, color='gray')
        y2 = np.linspace(slack[1,0],1,50)
        x2 = np.linspace(0,1-slack[1,0],50)
        ax.plot(x2, y2, color='gray')
        ax.fill_between(np.linspace(0,slack[0,1],20), np.zeros(20), np.linspace(slack[1,0],slack[0,1]+slack[1,0],20), facecolor='gray', alpha=0.2)
        ax.fill_between(np.linspace(1-slack[1,0],1,20), np.linspace(1-slack[0,1] - slack[1,0],1-slack[0,1],20), np.ones(20), facecolor='gray', alpha=0.2)
        ax.fill_between(np.linspace(slack[0,1],1-slack[1,0],20), np.linspace(0,1-slack[0,1]-slack[1,0],20), np.linspace(slack[0,1]+slack[1,0],1,20), facecolor='gray', alpha=0.2)
        ax.plot(sampled_x, sampled_y, '-^', label="Dec'n path")
        ax.plot([optimal_x[0]], [optimal_x[1]], 'r^', label='Optimum')
        ax.plot([x_const], [y_const], 'g^', label='Const. opt')
        ax.set_xlabel('Group 1 Decision', fontsize=10)
        ax.set_ylabel('Group 2 Decision', fontsize=10)
        plt.legend(loc='lower right')


        ax.set_title(f'Decision path with T = {T:,}')
        plt.savefig("fig6.png", dpi=350)
        plt.show()
        
    return price_paths, price_paths_types

# constants for defining the objective functions
root0 = .4
scale0 = max(root0,1-root0)**2
root1 = .9
scale1 = max(root1,1-root1)**2

# objective function for first group
def f_0(x):
    return (x-root0)**2/scale0

# objective function for second group
def f_1(x):
    return (x-root1)**2/scale1

# power series coefficients for f_0 and f_1
a1 = 1/scale0
b1 = -2*root0/scale0
c1 = root0**2 / scale0

a2 = 1/scale1
b2 = -2*root1/scale1
c2 = root1**2 / scale1

# define the slack 
slack = np.array([[0,.2], [.2,0]])

# find the constrained optimum
if root0 > root1 + slack[0,1]:
    lam = (1/(a1 + a2)) * (2 * a1 * a2 * slack[0,1] - a1 * b2 + a2 * b1)
    x_const = (lam - b1)/(2 * a1)
    y_const = (-lam - b2)/(2 * a2)
elif root1 > root0 + slack[1,0]:
    lam = (1/(a1 + a2)) * (-2 * a1 * a2 * slack[1,0] - a1 * b2 + a2 * b1)
    x_const = (lam - b1)/(2 * a1)
    y_const = (-lam - b2)/(2 * a2)
else:
    x_const = root0
    y_const = root1

# run SCAda-LGD
DynamicLaggedGD_2G(T,delta_naught,.7,std_noise,n_adjustment,n_min,1,0,1,f_0,f_1,slack,runs,gradient_threshold,[x_const,y_const],jump_adjustment)