import numpy as np

class ll_controler():
    def __init__(self):
        self.error_prev = 0
        self.error_int = 0

    def pid(self, set, position, velocity):
        error = set - position
        self.error_int += error
        #velocity = - velocity
        velocity = error - self.error_prev
        self.error_prev = error
        k_p = 20 #20
        k_d = 120 #120
        k_i = 0 #0

        u = k_p*error + k_d*velocity + k_i*self.error_int
        u = np.clip(u,-1,1)
        return u

    def bangbang(self, set, position):
        error = set - position
        tolerance = 0.01
        if error > tolerance:
            u = 1
        elif error < -tolerance:
            u = -1
        else:
            u = 0
        return u
