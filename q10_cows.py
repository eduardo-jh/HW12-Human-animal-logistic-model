#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW12 - Question 10. Suppose a ranch can sustain a herd of cattle and that the
cattle population can be modeled by the logistic equation:
    dP/dt=0.001*(500-P)*P
where dP/dt is the rate of change in the number of the heard per year so the
units for the rate constant are: /year.
a) Graph dP/dt vs P and determine the equilibrium states of this base model
   and label which are stable an unstable models.
b) If we choose to sell one animal per week (52/year), what is the new model?
c) Graph P vs t for both models, statting with 2 cows

Created on Tue Feb 23 12:27:06 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 1  # time step in years
P0 = 2
steps = 30
sell = 52 # one cow per week, in one year
r = 0.001
K = 500

t = np.array(range(0, steps+1, dt))
P = np.zeros(len(t))
Psell = np.zeros(len(t))
dPdt = np.zeros(len(t))
P[0], Psell[0], dPdt[0] = P0, P0, P0

for i in range(1, len(t)):
    # dPdt[i] = 0.001*(500 - P[i-1])*P[i-1]
    dPdt[i] = r*(1 - P[i-1]/K)*P[i-1]
    P[i] = P[i-1] + dPdt[i]*dt
    Psell[i] = P[i] if P[i] < sell else P[i] - sell

plt.figure(0)
plt.plot(t, P, 'b-')
plt.plot(t, Psell, 'r--')
plt.xlabel('Years')
plt.ylabel('Cows')

plt.figure(1)
plt.plot(P, dPdt, 'b-')
plt.xlabel('Years')
plt.ylabel('Change (dP/dt)')