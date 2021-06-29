# Analytical computations
f, ax = plt.subplots(1,1)
f.suptitle('G* function with variable coefficients')

# import modules
import numpy as np

def Vi(ai,alphai):
    return alphai**2/((1+2*alphai)*(1+ai)**2)

def V(a_prms,alpha):
    D=1
    for ai,alphai in zip(a_prms,alpha):
        D*=(1+Vi(ai,alphai))     
    return D-1


def S_i(a,alpha):
    S_i=np.zeros_like(a)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_i[i]=Vi(ai,alphai)/V(a,alpha)
    return S_i

def S_T(a,alpha):
    # to be completed
    S_T=np.zeros_like(a)
    Vtot=V(a,alpha)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_T[i]=(Vtot+1)/(Vi(ai,alphai)+1)*Vi(ai,alphai)/Vtot
    return S_T


def update_Sobol(**kwargs):
    import re
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    ax.clear()
    prm_cat=int(len(kwargs)/k)
    prms=np.zeros((prm_cat,k))
 
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post=r.match(key).groups()
        cat_idx=strings.index(pre)
        prms[cat_idx,int(post)]=value
            
        
    Si[:]=S_i(prms[0,:],prms[1,:])
    ST[:]=S_T(prms[0,:],prms[1,:])
    width=0.4
    x_tick_list=np.arange(len(prms[0,:]))+1
    ax.set_xticks(x_tick_list+width/2)
    x_labels=['x'+str(i) for i in np.arange(len(prms[0,:]))]
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0,1)

    ax.bar(x_tick_list,Si,width,color='blue')
    ax.bar(x_tick_list+width,ST,width,color='red')        
    ax.legend(['First order indices','Total indices'])

k=4 #number of prms
strings=['a','alpha','delta']
a_lbls=[strings[0]+str(i) for i in np.arange(k)]
alpha_lbls=[strings[1]+str(i) for i in np.arange(k)]
delta_lbls=[strings[2]+str(i) for i in np.arange(k)]
Si=np.zeros(k)
ST=np.zeros(k)
a_prms=np.zeros(k)
alpha=np.zeros_like(a_prms)
delta=np.zeros_like(a_prms)



import ipywidgets as widgets    
my_sliders=[]
for i in range(k):
    my_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))
    my_sliders.append(widgets.FloatSlider(min=0, max=15, value=1.0, description=alpha_lbls[i]))
    my_sliders.append(widgets.FloatSlider(min=0, max=1.0, value=0.5, description=delta_lbls[i]))


slider_dict = {slider.description:slider for slider in my_sliders}
ui_left = widgets.VBox(my_sliders[0::3]) 
ui_mid  = widgets.VBox(my_sliders[1::3])
ui_right = widgets.VBox(my_sliders[2::3])
ui=widgets.HBox([ui_left,ui_mid,ui_right])


out=widgets.interactive_output(update_Sobol, slider_dict) 

display(ui,out)

# End Analytical computations
