# Analytical computations
f, ax = plt.subplots(1,1)
f.suptitle('G function variable a coefficients')

# import modules
import numpy as np

def Vi(ai):
    return 1/(3*(1+ai)**2)

def V(a_prms):
    D=1
    for a in a_prms:
        D*=(1+Vi(a))     
    return D-1

def S_i(ai,a):
    return Vi(ai)/V(a)

def S_T(ai,a):
    Dtot=V(a)
    return (Dtot+1)/(Vi(ai)+1)*Vi(ai)/Dtot

def update_Sobol(**kwargs):
    ax.clear()
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post = key.split("a")
        assert pre==""
        a_prms[int(post)] = value
    
    width=0.4
    x_tick_list=np.arange(len(a_prms))+1
    ax.set_xticks(x_tick_list+width/2)
    x_labels=['x'+str(i) for i in np.arange(len(a_prms))]
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0,1)
    
    for i, a in enumerate(a_prms):
        Si[i]=S_i(a,a_prms)
        ST[i]=S_T(a,a_prms)
        
    ax.bar(x_tick_list,Si,width,color='blue')
    ax.bar(x_tick_list+width,ST,width,color='red')        
    ax.legend(['First order indices','Total indices'])
      
k=4 #number of prms
a_lbls=['a'+str(i) for i in np.arange(k)]
Si=np.zeros(k)
ST=np.zeros(k)
a_prms=np.zeros(k)

import ipywidgets as widgets    
my_sliders=[]
for i in range(k):
    my_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))


slider_dict = {slider.description:slider for slider in my_sliders}
ui_left = widgets.VBox(my_sliders[0::2]) 
ui_right = widgets.VBox(my_sliders[1::2])
ui=widgets.HBox([ui_left,ui_right])

out=widgets.interactive_output(update_Sobol, slider_dict) 
display(ui,out)


# End Analytical computations
