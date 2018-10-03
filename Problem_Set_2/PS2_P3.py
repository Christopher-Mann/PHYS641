# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:29:45 2018

@author: Chris
"""


from __future__ import division
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from math import pi



n=100

x = np.linspace(-1,1,n)


# Building the Noise matrix
N = np.ones([n,n]) + np.eye(n)
# Find the eigenvectors & eigenvalues
w,v = np.linalg.eigh(N)# w = Lambda, v = v
w = np.real(w); v=np.real(v)

# As worked out during office hours
"""
Chi^2 = (d-m)^T N^-1 (d-m)
      =   n^T N^-1 n                -> n is noise for correct model
      = n^T (v L v^T)^-1 n          -> decompose N into eigenvectors/values
                                       v is vectors, L is diagonal matrix of eigenvalues
      = n^T (v^T)^-1 L^-1 v^-1 n
      = n^T v L^-1 v^T n            -> Note: (v^T)^-1 = v, and v^T = v^-1
      = b^T L^-1 b                  -> b = v^T n  (b = n-squiggle in class)
      
      Since L is diagonal, b gives us a form of "data" in this chi^2 formulation
      convert b to n to get "correct" data
"""
dat_squig = np.random.normal(scale=np.sqrt(w))
dat = np.dot(v,dat_squig)

plt.figure(num=31)
plt.clf()
plt.plot(dat,'k.')
plt.ylim(-5,5)
plt.savefig("PS2_P3_single_realization.png")




# do many realizations of generated data
# test to see if <dd^T> looks like N 

niters=10000
ddT = np.zeros([n,n])
for i in range(niters):
    dat_squig_i = np.random.normal(scale=np.sqrt(w))
    dat_i = np.dot(v,dat_squig_i)
    
    ddT = ddT + np.outer(dat_i,dat_i)

ddTavg = ddT/niters
# sanity check
print N[:4,:4]
print ddTavg[:4,:4]
    

plt.figure(num=32)
plt.clf()
plt.title("<ddT>")
plt.imshow(ddTavg,interpolation='nearest',vmin=0,vmax=3)
plt.colorbar()
plt.savefig("PS2_P3_<ddT>.png")

plt.figure(num=33)
plt.clf()
plt.title("N")
plt.imshow(N,interpolation='nearest',vmin=0,vmax=3)
plt.colorbar()
plt.savefig("PS2_P3_N_matrix.png")

"""
The <ddT> and N matrices look very similar.  With 10000 iterations in the average
there is a small scatter around the N matrix values (see figures)

"""








