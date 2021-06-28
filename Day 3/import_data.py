'''
Created on Nov 18, 2016

@author: leifh
'''
import scipy.io as sio
import numpy as np

def import_data(filename):

    data = sio.loadmat(filename)
    q = data["q"]
    p = data["p"]
    p=p.flatten()  # convert array to vector
    q=q.flatten()
    N = len(q)
    
    if (filename=='jts_mlab.mat'):
        t= data["t"]
        t=t.flatten()
        dt = t[1]-t[0]
    else:
        dt=data['dt']    
        t=np.transpose(np.arange(N)*dt) #construct time vector from N and dt
        t=t.flatten()
        
    return  p, q, t, dt, N
