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
from math import pi,e











numpoints = 1000
x = np.linspace(-1,1,numpoints)
y_true = 4*x**3 - 1*x**2 - 2*x + 2
noise = 0.5
y = y_true + noise*np.random.randn(numpoints)



plt.figure(num=1)
plt.clf()
plt.title("Poly-fits: Red=QR, Blue=Classic")
plt.ylim(-2,4)
plt.plot(x,y_true,'m--',lw=3,label="True")
plt.plot(x,y,'k.',ms=4,label="Data")


plt.figure(num=12)
plt.clf()
plt.subplot(2,1,1)
plt.title("Residuals: Predicted - True")
plt.ylim(-5,5)
plt.ylabel("Classic")
plt.subplot(2,1,2)
plt.ylim(-5,5)
plt.ylabel("QR")



order = np.zeros(100)
rms1 = np.zeros(100)
rms2 = np.zeros(100)


iter = 0
for ord in range(0,50):
    
    A = np.zeros([numpoints,ord+1])
    A[:,0] = 1.0
    
    for i in range(ord):
        A[:,i+1] = A[:,i]*x
    N = np.identity(len(x))
    
    
    # doing QR decomposition
    Q,R = np.linalg.qr(A)
    Rinv = np.linalg.inv(R)
    QT = np.transpose(Q)
 
    """
    IF I WERE GOING TO USE GENERAL NOISE MATRIX...
    A^T N^-1 A m = A^T N^-1 d
       -> substitute A = QR
    R^T Q^T N^-1 Q R m = R^T Q^T N^-1 d
       -> Know that R and R^T are invertible 
       -> so multiply on the left by (R^T)^-1 left with:
    Q^T N^-1 Q R m = Q^T N^-1 d
       -> RESULT:
    m = (Q^T N^-1 Q R)^-1 Q^T N^-1 d
       -> if N = I, simplifies to 
    m = R^-1 Q^T d
    """
    
    RinvQT = np.dot(Rinv,QT)
    mQR = np.dot(RinvQT,y)

    predQR = np.dot(A,mQR)
    
    rmsQR = np.std(predQR-y_true)



    # doing standard ATNiA
    AT = np.transpose(A)
    Ni = np.linalg.inv(N)
    ATNi = np.dot(AT,Ni)
    ATNiA = np.dot(ATNi,A)
    
    LLS_matrix = np.dot(np.linalg.inv(ATNiA), ATNi)
    m = np.dot(LLS_matrix, y)
    
    pred_classic = np.dot(A,m)
    rms_classic = np.std(pred_classic-y_true)
    

    order[iter] = ord
    rms1[iter] = rmsQR
    rms2[iter] = rms_classic


    iter +=1

    plt.figure(num=1)
    plt.plot(x,predQR,'r--',label="Order %i, rms=%.4f"%(ord,rmsQR))
    plt.plot(x,pred_classic,'b-',label="Order %i, rms=%.4f"%(ord,rms_classic))
    plt.savefig("PS1_P1_QR_and_classic_fits.png")


    plt.figure(num=12)
    plt.subplot(2,1,1)
    plt.plot(x,pred_classic-y_true)
    plt.subplot(2,1,2)
    plt.plot(x,predQR-y_true)

plt.savefig("PS2_P1_QR_and_classic_residuals.png")












