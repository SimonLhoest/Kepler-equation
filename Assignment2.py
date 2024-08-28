# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:12:09 2023

@author: lhoes
"""

import numpy as np
import matplotlib.pyplot as plt
global e
global M
e=0
M=0
# For removing errors
deg2rad=np.pi/180
precision = 10**-15
max_iter=10
num=1001
ecc=np.linspace(0,1,num=num)
Mea=np.linspace(0,360,num=num)
                
def f(E):
    return E-e*np.sin(E)-M

def fp(E):
    return 1-e*np.cos(E)

def fpp(E):
    return e*np.sin(E)

def newton(E0, precision=1e-15, max_iter=max_iter):
    n=0
    while n<max_iter :
        n+=1
        E1=E0-f(E0)/fp(E0)
        if np.abs(E1-E0)<precision :
            break
        E0=E1
    return E1,n

def laguerre(E0, precision=1e-15, max_iter=max_iter):
    n=0
    N=5
    while n<max_iter :
        n+=1
        fE0=f(E0)
        fpE0=fp(E0)
        fppE0=fpp(E0)
        if (N-1)*fpE0<0:
            sign=-1
        else :
            sign=1
        delta = N*fE0/(fpE0 + sign*((N-1)**2*fpE0**2-N*(N-1)*fE0*fppE0)**(1/2))
        E1 = E0-delta
        if np.abs(E1-E0)<precision:
            break
        E0 = E1
    return E1,n

def boucle():
    global e 
    global M
    Newton   = np.zeros((len(Mea),len(ecc)))
    Laguerre = np.zeros((len(Mea),len(ecc)))
    nNewton  = np.zeros((len(Mea),len(ecc)))
    nLaguerre= np.zeros((len(Mea),len(ecc)))
    for Mi in range(len(Mea)):
        for ei in range(len(ecc)):
            M = Mea[Mi] * deg2rad
            e = ecc[ei]
            
            #First gues :
            E0 = M + e*np.cos(M)
            
            Newton[Mi,ei],nNewton[Mi,ei] = newton(E0,max_iter=10)
            Laguerre[Mi, ei],nLaguerre[Mi, ei] = laguerre(E0,max_iter=10)
    return Newton, nNewton, Laguerre, nLaguerre

N, nN, L ,nL = boucle()

#%%

plt.figure()
plt.title("Laguerre-Conway method with E0 = M * e * cos(M)")
plt.imshow(np.flipud(nL), cmap = 'plasma',  extent=[0, 10, 0, 10])
plt.colorbar(label='Iterations')
plt.ylabel("Mean anomaly M (°)")
plt.xticks(np.arange(11), labels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xlabel("Eccentricity e")
plt.yticks(np.arange(0,11,2.5), labels = [0,90,180,270,360])
plt.show()

plt.figure()
plt.title("Newton-Raphson method with E0 = M * e * cos(M)")
plt.imshow(np.flipud(nN), cmap = 'plasma',  extent=[0, 10, 0, 10])
plt.colorbar(label='Iterations')
plt.ylabel("Mean anomaly M (°)")
plt.xticks(np.arange(11), labels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xlabel("Eccentricity e")
plt.yticks(np.arange(0,11,2.5), labels = [0,90,180,270,360])
plt.show()


#%%
M = 300*deg2rad
e = 0.8
E0 = M + e*np.cos(M)

def newton(E0, precision=1e-15, max_iter=max_iter):
    n=0
    v=[]
    while n<max_iter :
        v.append(E0)
        n+=1
        E1=E0-f(E0)/fp(E0)
        if np.abs(E1-E0)<precision :
            break
        E0=E1
    v.append(E1)
    return v,n

def laguerre(E0, precision=1e-15, max_iter=max_iter):
    n=0
    N=5
    v=[]
    while n<max_iter :
        v.append(E0)
        n+=1
        fE0=f(E0)
        fpE0=fp(E0)
        fppE0=fpp(E0)
        if (N-1)*fpE0<0:
            sign=-1
        else :
            sign=1
        delta = N*fE0/(fpE0 + sign*((N-1)**2*fpE0**2-N*(N-1)*fE0*fppE0)**(1/2))
        E1 = E0-delta
        if np.abs(E1-E0)<precision:
            break
        E0 = E1
    v.append(E1)
    return v,n

u,n = newton(E0)
w,m = laguerre(E0)
u,w = np.array(u), np.array(w)

plt.figure()
plt.grid()
plt.title(f'Convergence for e={e}, M={M*1/deg2rad}')
plt.xlabel('Iterations')
plt.ylabel('E (degree)')
plt.plot(np.arange(n+1),u*1/deg2rad,'-o',label='Newton-Raphson')
plt.plot(np.arange(m+1),w*1/deg2rad,'-o',label='Laguerre-Conway')
plt.legend()
plt.xlim(0,max([n,m])+0.5)
plt.ylim(0,360)
plt.show()

