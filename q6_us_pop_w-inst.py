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
import statsmodels.api as sm
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

dPop = [pop[i]-pop[i-1] for i in range(1, len(pop))]  # increment between time steps

# Perform a linear regression with pop and dPop, then plot (dPop vs pop)
model = sm.OLS(dPop, pop[1:])  # No intercept by default, force through the origin as desired
results = model.fit()
r = results.params[0]  # growth rate (r) is the slope
print("*** Linear regression - numerical sol. (avg. growth rate) ***")
print("r=", r)

# Calculate doubling time and create an equation for exponential growth
tdouble = np.log(2)/np.log(1+r)*dt  # time to double population
mu1 = np.log(2)/tdouble  # constant for exponential eq. with base on natural log
print('tdouble =', tdouble, 'mu1=', mu1)
# Predict population with the model: P(t+1) = P[0]*(r+1)^t (numerical solution)
Pmodel = logistic_model(mu1, pop[0], t-start)

# Figure 0, plotting dPop vs pop
plt.figure(0)
plt.plot(pop[1:], dPop, 'bx', pop[1:], r*pop[1:], 'r-')  # plot and linear eq.
plt.legend(['data', 'linear regression $R^2$=%.4f' % results.rsquared], loc='best')
plt.xlabel('Population (P)')
plt.ylabel('Change populatiton (dPop)')
# plt.savefig('q6_us_pop_linear.png', dpi=300, bbox_inches='tight')

# Use 'minimize' to find the growth coefficient, mu
x0 = [0.1]  # initial guess for mu
res = minimize(sum_squared_error, x0, args=(t-start, pop))
mu2 = res.x[0]  # get the results
# mu2 *= 10  # anual to decade growth coefficient
print("\n*** Least squares - analytical sol. (inst. growth rate) ***")
print('mu2=', mu2)

# Predict population with exponential equation (analytical solution)
Pana = logistic_model(mu2, pop[0], t-start)

# Predict population using the Euler method (numerical solution)
ts = 0.5 # time step, years
end = int(start+dt*(len(t)-1))
steps = int((end-start)/ts)
time = np.linspace(start, end, steps)
Pnum = np.zeros(len(time))
Pnum[0] = pop[0]
for i in range(1, len(time)):
    Pnum[i] = Pnum[i-1] + mu2*Pnum[i-1]*ts

# Figure 1, plotting population growth vs t
plt.figure(1)
plt.plot(t, pop, 'bx', label=r'Pop (data)')
plt.plot(time, Pnum, 'r+', label=r'Pop (Euler dt=%.1f yr)' % ts)
plt.plot(t, Pmodel, 'r--', label=r'Pop (Avg growth rate)')
plt.plot(t, Pana, 'c-', label=r'Pop (ana) P=%.2f $\times$ $\exp$(%.3f $\times$ t)' % (pop[0], mu2))
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('US Population (millions)')
# plt.savefig('q6_us_pop.png', dpi=300, bbox_inches='tight')
