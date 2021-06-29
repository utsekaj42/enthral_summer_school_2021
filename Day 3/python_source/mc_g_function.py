'''
Created on May 12, 2018
@author: leifh
'''

# model function
import numpy as np
from present_output import print_vectors_relerror
import matplotlib.pyplot as plt

def g(Xj,aj):
    return (np.abs(4*Xj-2.)+aj)/(1+aj)


def G(X,a):
    G_vector=np.ones(X.shape[0])
    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[:,j],aj),G_vector)
    return G_vector

# end model function

import chaospy as cp

# Si with monte carlo for G-function

import monte_carlo as mc
k= 8
a_prms=np.ones(k)
a_prms[0] = 10
a_prms[1] = 10
cp.seed(0)
jpdf = cp.Iid(cp.Uniform(),k)
Ns=5000
print('Number of samples for Monte Carlo: ', Ns) 
X=jpdf.sample(Ns)
A, B, C = mc.generate_sample_matrices(Ns, jpdf, sample_method='R') #A, B, C already transposed
G_A_sample = G(A, a_prms)
G_B_sample = G(B, a_prms)
G_C_sample_list = np.array([G(C_i, a_prms) for C_i in C])
print(A.shape, B.shape, C.shape)
print(G_A_sample.shape, G_C_sample_list.shape)
exp_mc = np.mean(G_A_sample)
std_mc = np.std(G_A_sample)
print("Statistics Monte Carlo\n")
print('\n        E(Y)  |  std(Y) \n')
print('mc  : {:2.5f} | {:2.5f}'.format(float(exp_mc), std_mc))

S_mc, S_tmc = mc.calculate_sensitivity_indices(G_A_sample, G_B_sample, G_C_sample_list)
row_labels= ['S_'+str(idx) for idx in range(k)]
col_labels=['Monte carlo','Analytical','Error (%)']

print("\nFirst Order Indices")
import analytical_g_function as agf

Si=np.zeros(k)
ST=np.zeros(k)
for i, a in enumerate(a_prms):
    Si[i]=agf.S_i(a,a_prms)
    ST[i]=agf.S_T(a,a_prms)
 
print_vectors_relerror(S_mc, Si, col_labels, row_labels, [3,3,0])

print("\n\nTotal Effect Indices")
row_labels= ['St_'+str(idx) for idx in range(k)]
print_vectors_relerror(S_tmc, ST, col_labels, row_labels, [3,3,0])
# end Si with monte carlo for G-function
