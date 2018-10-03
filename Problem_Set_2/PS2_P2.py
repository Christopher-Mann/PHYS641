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
    PART A)    
"""

print "Part a) \n  Fitting Chebyshev polynomials to exp(x) from [-1,1]"

numpoints = 1000
x = np.linspace(-1,1,numpoints)
y_true = np.exp(x)
noise = 0.5
y = y_true #+ noise*np.random.randn(numpoints)



plt.figure(num=21)
plt.clf()

plt.ylim(-1,4)



minord = 1
maxord = 28



for ord in range(minord,maxord):
    
    A = np.zeros([numpoints,ord+1])
    A[:,0] = 1.0
    A[:,1] = x
    
    for i in np.arange(1,ord):
        A[:,i+1] = 2*x*A[:,i] - A[:,i-1]

    N = np.identity(len(x))
    
#    # doing standard ATNiA
#    AT = np.transpose(A)
#    Ni = np.linalg.inv(N)
#    ATNi = np.dot(AT,Ni)
#    ATNiA = np.dot(ATNi,A)  #   
#    
#    LLS_matrix = np.dot(np.linalg.inv(ATNiA), ATNi)
#    m = np.dot(LLS_matrix, y)
#    
#    pred_classic = np.dot(A,m)
#    rms_classic = np.std(pred_classic-y_true)
    
    # doing QR decomposition
    # USING THE SAME SIMPLIFICATION SHOWN IN PROBLEM 1 REGARDING N=I 
    Q,R = np.linalg.qr(A)
    Rinv = np.linalg.inv(R)
    QT = np.transpose(Q)
    
    RinvQT = np.dot(Rinv,QT)
    mQR = np.dot(RinvQT,y)

    predQR = np.dot(A,mQR)
    
    rmsQR = np.std(predQR-y_true)
    
    
    plt.plot(x,predQR,'-',lw=2)#,label="Order %i, rms=%.4f"%(ord,rmsQR))
#    plt.plot(x,pred_classic,'-',label="Order %i, rms=%.4f"%(ord,rms_classic))

plt.legend(fontsize=9,ncol=3,loc='center bottom')
plt.plot(x,y,'k--',lw=2,label="Data")
plt.text(-0.8,3,"Plotted order %i to %i"%(minord,maxord))







#'''
"""
    PART B)    
"""

ord = 6

print "Part b) \n ** Fitting 6th order Chebyshev to exp(x) from [-1,1]"

A = np.zeros([numpoints,ord+1])
A[:,0] = 1.0
A[:,1] = x

for i in np.arange(1,ord):
    A[:,i+1] = 2*x*A[:,i] - A[:,i-1]

N = np.identity(len(x))

# doing QR decomposition
# USING THE SAME SIMPLIFICATION SHOWN IN PROBLEM 1 REGARDING N=I 
Q,R = np.linalg.qr(A)
Rinv = np.linalg.inv(R)
QT = np.transpose(Q)

RinvQT = np.dot(Rinv,QT)
mQR = np.dot(RinvQT,y)
#print mQR[:10]

predQR = np.dot(A,mQR)
rmsQR = np.std(predQR-y_true)

rms7 =  np.std(predQR-y_true)       # for a 7-term fit
max7 = np.max(np.abs(predQR-y_true))


print "RMS error = %.4e"%(rmsQR)
print "Max error = %.4e"%(np.sum(np.abs(mQR)))
    
#'''

ord = 200
print "Part b) \n ** Truncating order %i fit to 7 terms"%(ord)

A = np.zeros([numpoints,ord+1])
A[:,0] = 1.0
A[:,1] = x



for i in np.arange(1,ord):
    A[:,i+1] = 2*x*A[:,i] - A[:,i-1]

N = np.identity(len(x))

# doing QR decomposition
Q,R = np.linalg.qr(A)
Rinv = np.linalg.inv(R)
QT = np.transpose(Q)

RinvQT = np.dot(Rinv,QT)
mQR = np.dot(RinvQT,y)

predQR = np.dot(A,mQR)



"""
do all the fitting with the full high-order polynomial
Truncate the A matrix and m vector to only have 7 columns or terms, respectively
"""
A_trunc = A[:,:7]
mQR_trunc = mQR[:7]


predQR_trunc = np.dot(A_trunc,mQR_trunc)




errRMS7 = rms7  # from 7-term fit
errMAX7 = max7

errRMS_trunc = np.std(predQR_trunc-y_true)  # from order 200 fit, truncated to 7 terms
errMAX_trunc = np.max(np.abs(predQR_trunc - y_true)) 
errMAX_trunc_predicted = np.sum(np.abs(mQR[7:]))

print "Errors:"
print "RMS error 7-term fit \n    %.4e"%(errRMS7)
print "Max error 7-term fit \n    %.4e"%(errMAX7)
    
print "RMS error using 1st 7 terms of order 200 fit \n    %.4e"%(errRMS_trunc)
print "Max error using 1st 7 terms of order 200 fit \n    %.4e"%(errMAX_trunc)# ???
print "Max error expected summing coefficients 8 to %i \n    %.4e"%(ord,errMAX_trunc_predicted)
    
print "RMS error increases by factor: %.3f"%(errRMS7/errRMS_trunc)**-1
print "Max error decreases by factor: %.3f"%(errMAX7/errMAX_trunc)**-1

""" 
This outputs:

Errors:
RMS error 7-term fit 
    1.9853e-06
Max error 7-term fit 
    7.9848e-06
RMS error using 1st 7 terms of order 200 fit 
    2.2588e-06
Max error using 1st 7 terms of order 200 fit 
    3.4094e-06
Max error expected summing coefficients 8 to 200 
    3.4148e-06
RMS error increases by factor: 1.138
Max error decreases by factor: 0.427

"""



    
