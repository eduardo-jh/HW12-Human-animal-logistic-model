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
   and label which are stable and unstable models.
b) If we choose to sell one animal per week (52/year), what is the new model?
c) Graph P vs t for both models, statting with 2 cows

Created on Tue Feb 23 12:27:06 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 1  # time step in years
P0 = 150
steps = 100
sell = 52  # one cow per week, in one year=52
r = 0.001  # growth rate 
K = 500  # carrying capacity

t = np.array(range(0, steps+1, dt))
P = np.zeros(len(t))
Psell = np.zeros(len(t))
P[0], Psell[0] = P0, P0

dPselldt = np.zeros(len(t))
dPdt = np.zeros(len(t))
dPdt2 = np.zeros(K+1)

for i in range(1, len(t)):
    # No sell
    dPdt[i] = r*(K - P[i-1])*P[i-1]
    P[i] = P[i-1] + dPdt[i]*dt
    # Selling cows
    dPselldt[i] = r*(K - Psell[i-1])*Psell[i-1]
    Psell[i] = Psell[i-1] + dPselldt[i]*dt - sell

for i in range(1, K):
    dPdt2[i] = r*(K-i)*i - sell

plt.figure(0)
plt.plot(t, P, 'b-', label='Cows')
plt.plot(t, Psell, 'r--', label='Cows selling %d/yr' % sell)
plt.legend(loc='best')
plt.xlabel('Years')
plt.ylabel('Cows')
plt.grid()
plt.savefig('q10_cows.png', dpi=300, bbox_inches='tight')

plt.figure(1)
plt.plot(range(K+1), dPdt2, 'b-')
plt.xlabel('K')
plt.ylabel('dP/dt2')
plt.grid()
plt.savefig('q10_cows_dPdt2.png', dpi=300, bbox_inches='tight')