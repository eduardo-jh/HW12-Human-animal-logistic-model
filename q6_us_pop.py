#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW12 - Question 6. US population growth from 1800 to 1860

Created on Tue Feb 23 12:27:06 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def logistic_model(mu, p0, t):
    """ The analytical solution for the logistic model as population growth. 

    mu: float, the growth coefficient.
    p0: int, initial population.
    t: array, the time step vector.
    Returns: array, a population value for each time step.
    """
    return p0 * np.exp(mu*t)
    
def sum_squared_error(x, t, P):
    """ Sum of squared errors
    
    x: float, the search parameter
    t: array, the time step vector.
    P: array, data array to compare against predictions.
    return: sum of squared errors
    """
    assert len(t) == len (P), "Length of t and P are not equal"
    Pop = logistic_model(x, P[0], t)  # predictions with logistic model (analytical)
    SS = pow(Pop - P, 2)  # squared errors between analytical and numerical solutions
    SSE = sum(SS)
    return SSE

start = 1800  # start year
dt = 10  # time step, years
pop = np.array([5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4])  # in millions
t = np.array(range(start, start+(dt*len(pop)), dt))  # time vector, years

# Use 'minimize' to find the growth coefficient, mu
x0 = [0.1]  # initial guess for mu
res = minimize(sum_squared_error, x0, args=(t-start, pop))
mu = res.x[0]  # get the results
# mu *= 10  # anual to decade growth coefficient
print("\n*** Least squares - analytical sol. (inst. growth rate) ***")
print('mu=', mu)

# Predict population with exponential equation (analytical solution)
Pana = logistic_model(mu, pop[0], t-start)

# Predict population using the Euler method (numerical solution)
ts = 0.5 # time step, years
end = int(start+dt*(len(t)-1))
steps = int((end-start)/ts)
time = np.linspace(start, end, steps)
Pnum = np.zeros(len(time))
Pnum[0] = pop[0]
for i in range(1, len(time)):
    Pnum[i] = Pnum[i-1] + mu*Pnum[i-1]*ts

# Figure 1, plotting population growth vs t
plt.figure(1)
plt.plot(t, pop, 'bx', label=r'Pop (data)')
plt.plot(time, Pnum, 'r+', label=r'Pop (Euler dt=%.1f yr)' % ts)
plt.plot(t, Pana, 'c-', label=r'Pop (ana) P=%.2f $\times$ $\exp$(%.3f $\times$ t)' % (pop[0], mu))
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('US Population (millions)')
# plt.savefig('q6_us_pop.png', dpi=300, bbox_inches='tight')
