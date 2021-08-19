import numpy as np

class ll_controler():
    def __init__(self):
        self.error_prev = 0
        self.error_int = 0

    def pd(self, set, position, velocity):
        error = set - position
        self.error_int += error
        #velocity = - velocity
        velocity = error - self.error_prev
        self.error_prev = error
        k_p = 3 #1
        k_d = 100 #120
        k_i = 0 #0.001
        u = k_p*error + k_d*velocity + k_i*self.error_int
        u = np.clip(u,-1,1)
        return u

    def bangbang(self, set, position):
        error = set - position
        tolerance = 0.005
        if error > tolerance:
            u = 1
        elif error < -tolerance:
            u = -1
        else:
            u = 0
        return u
