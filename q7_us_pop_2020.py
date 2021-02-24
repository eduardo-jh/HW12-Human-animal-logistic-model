#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW12 - Question 7. US population growth from 1800 to 2020

Created on Tue Feb 23 23:28:18 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt

start = 1800  # start year
end = 2020
dt = 10  # time step, years
P0 = 5.3  # initial population
mu = 0.0295  # growth rate from Q6

# Estimate US population to 2020 assuming exponential growth
t = np.array(range(start, end+1, dt))  # time vector, years
Pana = P0 * np.exp(mu*(t-start))

# Estimate US population to 2020 using Euler approximation
ts = 1 # time step, years
steps = int((end-start)/ts)
time = np.linspace(start, end, steps)
Pnum = np.zeros(len(time))
Pnum[0] = P0
for i in range(1, len(time)):
    Pnum[i] = Pnum[i-1] + mu*Pnum[i-1]*ts

print("The US population in 2020: %.2f (%.2f approx.) million" % (Pana[-1], Pnum[-1]))

# Figure 1, plotting population growth vs t
plt.figure(1)
plt.plot(time, Pnum, 'r--', label=r'Pop (Euler dt=%.1f yr)' % ts)
plt.plot(t, Pana, 'c-', label=r'Pop (ana) P=%.2f $\times$ $\exp$(%.3f $\times$ t)' % (P0, mu))
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('US Population (millions)')
# plt.savefig('q7_us_pop_2020.png', dpi=300, bbox_inches='tight')