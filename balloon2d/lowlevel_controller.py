import numpy as np

def ll_pd(set,position,velocity):
    error = set - position
    velocity = - velocity
    k_p = 2
    k_d = 10
    u = k_p*error + k_d*velocity
    u = np.clip(u,-1,1)
    return u
