'''
Created on May 12, 2018
@author: leifh
'''

# model function
import numpy as np

def g(Xj,aj,alphaj,deltaj):
    return ((1+alphaj)*np.abs(2*(Xj+deltaj-(Xj+deltaj).astype(int))-1)**alphaj+aj)/(1+aj)


def G(X,a,alpha,d):
    G_vector=np.ones(X.shape[0])

    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[:,j],aj,alpha[j],d[j]),G_vector)
    return G_vector

# end model function



# scalar model function

def g(Xj,aj):
    return (np.abs(4*Xj-2.)+aj)/(1+aj)

def G_s(X,a):
    G_scalar=1.0
    
    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[j],aj),G_scalar)
    return G_scalar

# end model function

# Number of params:
k=4

# Si with chaospy for G-function
import chaospy as cp
import chaospy_wrapper as cpw

jpdf = cp.Iid(cp.Uniform(),k)
    
polynomial_order = 4
basis = cpw.generate_basis(polynomial_order, jpdf)

#Ns=2*len(basis['poly'])
Ns=500
print('Number of samples for chaospy: ', Ns) 
X=jpdf.sample(Ns)
G_sample=G(X.transpose(),a_prms)

expansion = cpw.fit_regression(basis, X, G_sample)

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

# chaospy G-function with sliders
import chaospy as cp
import chaospy_wrapper as cpw

if not 'jpdf' in globals():
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')


def update_chaospy_G(**kwargs):
    NS=kwargs['NS']
    del kwargs['NS']
    polynomial_order=kwargs['polynomial_order']
    del kwargs['polynomial_order']

    prm_cat=int(len(kwargs)/k)
    prms=np.zeros((prm_cat,k))

    import re
    r = re.compile("([a-zA-Z]+)([0-9]+)")

 
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post=r.match(key).groups()
        cat_idx=strings.index(pre)
        prms[cat_idx,int(post)]=value


    X=jpdf.sample(NS)
    print('Number of samples: ',NS)

    G_sample=G(X.transpose(),prms[0,:],prms[1,:],prms[2,:])

    basis = cpw.generate_basis(polynomial_order, jpdf)
    expansion = cpw.fit_regression(basis, X, G_sample)

    exp_pc = cpw.E(expansion, jpdf)
    std_pc = cpw.Std(expansion, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    S_pc = cpw.Sens_m(expansion, jpdf) #Si from chaospy
    S_tpc = cpw.Sens_t(expansion, jpdf) #Total effect sensitivity index from chaospy
    
    row_labels= ['S_'+str(idx) for idx in range(len(a_prms))]
    col_labels=['Chaospy','Analytical','Error (%)']

    print("\nFirst Order Indices")
    print_vectors_relerror(S_pc,Si,col_labels,row_labels,[3,3,0])

    print("\n\nTotal Effect Indices")
    row_labels= ['St_'+str(idx) for idx in range(k)]
    print_vectors_relerror(S_tpc,ST,col_labels,row_labels,[3,3,0])


if (len(my_sliders)==len(a_prms)*3):   #add sliders if not added before
    my_sliders.append(widgets.IntSlider(min=500,max=5000,step=200,value=500,description='NS')) #add slider for samples
    my_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

    slider_dict = {slider.description:slider for slider in my_sliders} #add the sliders in the dictionary 

    ui_left = widgets.VBox(my_sliders[0::3]) 
    ui_mid  = widgets.VBox(my_sliders[1::3])
    ui_right = widgets.VBox(my_sliders[2::3])
    ui=widgets.HBox([ui_left,ui_mid,ui_right])

out=widgets.interactive_output(update_chaospy_G, slider_dict) 
display(ui,out)

# end chaospy G-function with sliders




