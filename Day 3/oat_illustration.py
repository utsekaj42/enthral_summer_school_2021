'''
Created on Jun 23, 2017
@author: leifh
'''
import numpy as np
import matplotlib.pyplot as plt

# Module to illustrate the shortcomings of the One factor At the Time
# algorithm.
# To avoid the usage of the Gamma-function, a recursive 
# formulation was implemented. 
# See https://en.wikipedia.org/wiki/N-sphere#Recurrences

def hyperSphere_hyperCube_ratio(N):
    Vsphere=[1]
    Ssphere=[2]
    Vcube=[1]
    SphereCubeRatio=[]
        
    for n in range(0,N):
        Ssphere.append(2*np.pi*Vsphere[n])
        Vsphere.append(Ssphere[n]/(n+1))
        Vcube.append(2**(n+1))
        SphereCubeRatio.append(Vsphere[-1]/Vcube[-1])
        
    return SphereCubeRatio
        
Ndim=10
plt.plot(hyperSphere_hyperCube_ratio(Ndim))
_=plt.xlabel('Number of dimensions')
_=plt.ylabel('Hypersphere to hypercube volume ratio')
plt.show()

