'''
Created on May 12, 2018
@author: leifh
'''

# model function
import numpy as np

def g(Xj,aj):
    return (np.abs(4*Xj-2.)+aj)/(1+aj)


def G(X,a):
    G_vector=np.ones(X.shape[0])
    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[:,j],aj),G_vector)
    return G_vector

# end model function

# Number of params:
k=4

# Si with chaospy for G-function
import chaospy as cp
import chaospy_wrapper as cpw
    
jpdf = cp.Iid(cp.Uniform(),k)
    
polynomial_order = 4
poly = cp.generate_expansion(polynomial_order, jpdf)

#Ns=2*len(basis['poly'])
Ns=500
print('Number of samples for chaospy: ', Ns) 
Xpc=jpdf.sample(Ns)
G_sample=G(Xpc.transpose(),a_prms)
basis = cpw.generate_basis(polynomial_order, jpdf)
expansion = cpw.fit_regression(basis, Xpc, G_sample)

exp_pc = cpw.E(expansion, jpdf)
std_pc = cpw.Std(expansion, jpdf)
print("Statistics polynomial chaos\n")
print('\n        E(Y)  |  std(Y) \n')
print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
S_pc = cpw.Sens_m(expansion, jpdf) #Si from chaospy
S_tpc = cpw.Sens_t(expansion, jpdf) #Total effect sensitivity index from chaospy

row_labels= ['S_'+str(idx) for idx in range(k)]
col_labels=['Chaospy','Analytical','Error (%)']


print("\nFirst Order Indices")

print_vectors_relerror(S_pc,Si,col_labels,row_labels,[3,3,0])

print("\n\nTotal Effect Indices")
row_labels= ['St_'+str(idx) for idx in range(k)]
print_vectors_relerror(S_tpc,ST,col_labels,row_labels,[3,3,0])


# end Si with chaospy for G-function


# mc and pc comparison for G-function with sliders
import chaospy as cp
import chaospy_wrapper as cpw
import monte_carlo as mc
import analytical_g_function as agf

if not 'jpdf' in globals():
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')


def update_G(**kwargs):
    NsPC=kwargs['NsPC']
    del kwargs['NsPC']
    NsMC=kwargs['NsMC']
    del kwargs['NsMC']
    
    polynomial_order=kwargs['polynomial_order']
    del kwargs['polynomial_order']

    for key, value in kwargs.items():  # find indx and value for a_prms
        pre, post = key.split("a")
        assert pre == ""
        a_prms[int(post)] = value

    ## Update the analytical indices
    Si = np.zeros(k)
    ST = np.zeros(k)
    for i, a in enumerate(a_prms):
        Si[i] = agf.S_i(a, a_prms)
        ST[i] = agf.S_T(a, a_prms)

    ## Monte Carlo update
    print('Number of samples for Monte Carlo: ', NsMC)
    A, B, C = mc.generate_sample_matrices_mc(NsMC, k, jpdf, sample_method='R') #A, B, C already transposed
    G_A_sample = G(A, a_prms)
    G_B_sample = G(B, a_prms)
    G_C_sample_list = np.array([G(C_i, a_prms) for C_i in C])
    
    exp_mc = np.mean(G_A_sample)
    std_mc = np.std(G_A_sample)
    print("Statistics Monte Carlo\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('mc  : {:2.5f} | {:2.5f}'.format(float(exp_mc), std_mc))
    
    S_mc, S_tmc = mc.calculate_sensitivity_indices_mc(G_A_sample, G_B_sample, G_C_sample_list)


    ## update PC estimates
    Xpc=jpdf.sample(NsPC)
    print('Number of samples: ',NsPC)

    G_sample=G(Xpc.transpose(), a_prms)
    basis = cpw.generate_basis(polynomial_order, jpdf)
    expansion = cpw.fit_regression(basis, Xpc, G_sample)

    exp_pc = cpw.E(expansion, jpdf)
    std_pc = cpw.Std(expansion, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    S_pc = cpw.Sens_m(expansion, jpdf) #Si from chaospy
    S_tpc = cpw.Sens_t(expansion, jpdf) #Total effect sensitivity index from chaospy

    row_labels= ['S_'+str(idx) for idx in range(len(a_prms))]
    col_labels=['Monte Carlo','Err (%)','PolyChaos','Err (%)']


    print("\nFirst Order Indices")
    print_3vectors_relerror(S_mc,S_pc, Si, col_labels, row_labels, [3,0,3,0])

    print("\n\nTotal Effect Indices")
    row_labels= ['St_'+str(idx) for idx in range(k)]
    print_3vectors_relerror(S_tmc,S_tpc, ST, col_labels, row_labels, [3,0,3,0])



## Set up the sliders 
cmp_sliders=[]
for i in range(k):
    cmp_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))

cmp_sliders.append(widgets.IntSlider(min=500,max=5000,step=200,value=500,description='NsPC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(min=500,max=50000,step=200,value=500,description='NsMC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

slider_dict = {slider.description:slider for slider in cmp_sliders} #add the sliders in the dictionary 

ui_left = widgets.VBox(cmp_sliders[0::2])
ui_right = widgets.VBox(cmp_sliders[1::2])
ui=widgets.HBox([ui_left,ui_right])

out=widgets.interactive_output(update_G, slider_dict)
display(ui,out)

# end mc and pc comparison for G-function with sliders




