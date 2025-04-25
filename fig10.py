# -*- coding: utf-8 -*-
"""
Created on Jun 29 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file creates Figure 10 in "Algorithmic
Challenges in Ensuring Fairness at the Time 
of Decision", depicting gradient descent
trajectories given a non-separable objective
function.
"""


import numpy as np
import matplotlib.pyplot as plt
from random import *

# Parameters
N = 30

# Objective function, ax^2 + by^2 + cxy
a = 1
b = 1
c = 1

# Step size
eta = .1

# Display boundaries
x_min = -3
x_max = 3
y_min = -3
y_max = 3

# Initial points
x_initial = [-2.8,-2,-2,0,1.5]
y_initial = [-.28,1,-2,-2.4,-2]

# Make the figure
fig, ax = plt.subplots(figsize=(6.15,6.15)) 

def display_with_sign(f):
    """Formats coefficients for figure title"""
    if f == 1:
        return " $+$ "
    elif f == -1:
        return " $-$ "
    elif f >= 0:
        return " $+$ " + str(f)
    else:
        return " $-$ " + str(-f)
def display_without_sign(f):
    """Formats coefficients for figure title"""
    if f == 1:
        return ""
    elif f == -1:
        return " $-$ "
    elif f >= 0:
        return str(f)
    else:
        return " $-$ " + str(-f)

def cost_func(theta0, theta1):
    """The objective function, a*theta0^2 + 
    b*theta1^2 + c*theta0*theta1."""
    
    return a * theta0**2 + b * theta1**2 + c * theta0 * theta1

def gradient(theta_array):
    """Gradient of the objective function at the
    point theta_array."""
    ret = np.zeros(2)
    ret[0] = 2 * a * theta_array[0] + c * theta_array[1] 
    ret[1] = 2 * b * theta_array[1] + c * theta_array[0] 
    return ret

def constrained_gradient(theta_array):
    """Minimum of the gradient of the objective at 
    theta_array and 0."""
    ret = np.zeros(2)
    ret[0] = min(0,2 * a * theta_array[0] + c * theta_array[1]) 
    ret[1] = min(0,2 * b * theta_array[1] + c * theta_array[0]) 
    return ret

# x values and y values
theta0_grid = np.linspace(x_min,x_max,101)
theta1_grid = np.linspace(y_min,y_max,101)

# grid of coordinates
X, Y = np.meshgrid(theta0_grid, theta1_grid)

# draw level curves
J_grid = cost_func(X,Y)
contours = ax.contour(X, Y, J_grid, 30)
ax.clabel(contours)

nums = map(lambda x : random.randint(0,7), range(N))
base_colors = ['b', 'g', 'm', 'c', 'orange']
colors = []
for i in range(N):
    colors.append(base_colors[i%5])

# for each initial point...
for run in range(len(x_initial)):
    # Initial point for unconstrained gradient descent
    theta = [np.array((x_initial[run],y_initial[run]))]
    # Initial point for monotonic gradient descent
    theta_constrained = [np.array((x_initial[run],y_initial[run]))]
    
    # list of objective values
    J = [cost_func(*theta[0])]

    # run gradient descent
    for j in range(N-1):
        last_theta = theta[-1]
        last_theta_constrained = theta_constrained[-1]
        this_theta = np.empty((2,))
        this_theta_constrained = np.empty((2,))
        
        # Update point based on the negative gradient
        # (standard gradient descent update)
        this_theta = last_theta - eta * gradient(last_theta)
        
        # Update point based on the projection of the 
        # negative gradient on the positive orthant (gradient
        # descent variant with enforced monotonicity)
        this_theta_constrained = last_theta_constrained - eta * constrained_gradient(last_theta_constrained)

        # Add new iterates to our list of iterates
        theta.append(this_theta)
        theta_constrained.append(this_theta_constrained)
        J.append(cost_func(*this_theta))
    
    for j in range(1,N):
        # Plot the unconstrained trajectory
        ax.annotate('', xy=theta[j], xytext=theta[j-1],
                       arrowprops={'arrowstyle': '->', 'color': 'c', 'lw': 1},
                       va='center', ha='center')
    for j in range(1,N):
        # Plot the monotonicity-enforced trajectory
        ax.annotate('', xy=theta_constrained[j], xytext=theta_constrained[j-1],
                       arrowprops={'arrowstyle': '->', 'color': 'm', 'lw': 1},
                       va='center', ha='center')

ax.set_title('Decision path over level sets of ' + display_without_sign(a) + '$x(1)^2$' + display_with_sign(b) + '$x(2)^2$ ' + display_with_sign(c) + '$x(1)x(2)$')

plt.savefig("fig10.png", dpi=400)
plt.show()
