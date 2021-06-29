'''
Created on May 12, 2018
@author: leifh
'''

# model function
import numpy as np
from present_output import print_vectors_relerror
import matplotlib.pyplot as plt
import chaospy as cp

def g(Xj,aj):
    return (np.abs(4*Xj-2.)+aj)/(1+aj)


def G(X,a):
    G_vector=np.ones(X.shape[0])
    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[:,j],aj),G_vector)
    return G_vector

# end model function


#k=10#5 - looks good

# Si with monte carlo for G-function

import monte_carlo as mc
a_prms=np.ones(k)

if not 'jpdf' in globals():
    cp.seed(0)
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')

def update_mc_G(**kwargs):
    Ns=kwargs['NS']
    del kwargs['NS']
    
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post = key.split("a")
        assert pre==""
        a_prms[int(post)] = value
        


    print('Number of samples for Monte Carlo: ', Ns) 
    X=jpdf.sample(Ns)
    A, B, C = mc.generate_sample_matrices_mc(Ns, k, jpdf, sample_method='R') #A, B, C already transposed
    G_A_sample = G(A, a_prms)
    G_B_sample = G(B, a_prms)
    G_C_sample_list = np.array([G(C_i, a_prms) for C_i in C])
    
    exp_mc = np.mean(G_A_sample)
    std_mc = np.std(G_A_sample)
    print("Statistics Monte Carlo\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('mc  : {:2.5f} | {:2.5f}'.format(float(exp_mc), std_mc))
    
    S_mc, S_tmc = mc.calculate_sensitivity_indices_mc(G_A_sample, G_B_sample, G_C_sample_list)
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

## Set up the sliders 
mc_sliders=[]
for i in range(k):
    mc_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))

mc_sliders.append(widgets.IntSlider(min=500,max=25000,step=200,value=500,description='NS')) #add slider for samples
    
slider_dict = {slider.description:slider for slider in mc_sliders} #add the sliders in the dictionary 

ui_left = widgets.VBox(mc_sliders[0::2]) 
ui_right = widgets.VBox(mc_sliders[1::2])
ui=widgets.HBox([ui_left,ui_right])

out=widgets.interactive_output(update_mc_G, slider_dict) 
display(ui,out)

# end Si with monte carlo for G-function
