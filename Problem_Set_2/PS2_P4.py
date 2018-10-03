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

"""
    part a)
"""

x = np.arange(0,1000)

# Shape of the signal (amplitude = 1)
def G(x):
    return np.exp(-(x-500)**2 / (2*50**2))
    

# different correlation strenghths and lengths to explore
a = [0.1, 0.5, 0.9]
sig = [5, 50, 500]



# two loops for all combinations of [a,sigma]
for k in range(len(a)):
    for l in range(len(sig)):
        
        #build the noise matrix based on choice of [a,sigma]
        N = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(len(x)):
                N[i,j] = a[k]*np.exp( -(i-j)**2 / (2*sig[l]**2) )
        N = N + np.identity(len(x))*(1-a[k])
        
        # in a classical (A^T N^-1 A) least squares fit
        # sigma of m (amplitude) is 1/sqrt(A^T N^-1 A)
        A = G(x)
        AT = A.transpose()
        Ni = np.linalg.inv(N)
        ATNi = np.dot(AT,Ni)
        ATNiA = np.dot(ATNi,A)
        sig_amp = (ATNiA)**(-1/2)
    
        print "a=%.1f, sig=%3i -> amp uncertainty = %.3f"%(a[k],sig[l],sig_amp)

"""
Output:

a=0.1, sig=  5 -> amp uncertainty = 0.156
a=0.1, sig= 50 -> amp uncertainty = 0.338
a=0.1, sig=500 -> amp uncertainty = 0.128

a=0.5, sig=  5 -> amp uncertainty = 0.276
a=0.5, sig= 50 -> amp uncertainty = 0.714
a=0.5, sig=500 -> amp uncertainty = 0.101

a=0.9, sig=  5 -> amp uncertainty = 0.358
a=0.9, sig= 50 -> amp uncertainty = 0.950
a=0.9, sig=500 -> amp uncertainty = 0.049

"""



"""
    part b)

The worst error comes from a high correlation strenght (a) combined
with a moderate correlation length (sig).  I suspect this is problematic
because sizeable chunks of the data are tightly correlated, but not with
other sections of the data.  It's hard to tell how the system as a whole
is behaving.  This type of "clumpy" data would be very problematic.

The smallest error comes from a high correlation strength with a large correlation 
distance.  I suspect this is so low because sigma is comparable to the number
of data points in the sample (500 vs. 1000).  In this case most of the data set
is tightly correlated and therefore quite predictable.

The worst cases are consistently the "clumpy" ones where the correlation length
is bigger than just a few points, but not yet comparable to the number of data points.

"""




