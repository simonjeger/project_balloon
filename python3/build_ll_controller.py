import numpy as np

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class ll_controler():
    def __init__(self):
        self.error_prev = 0
        self.error_int = 0

        if yaml_p['balloon'] == 'indoor_balloon':
            if yaml_p['environment'] == 'python3':
                self.k_p = 2
                self.k_d = 80
                self.k_i = 0
            elif yaml_p['environment'] == 'vicon':
                self.k_p = 9.5
                self.k_d = 80
                self.k_i = 0
            else:
                self.k_p = 20 #20
                self.k_d = 120 #120
                self.k_i = 0 #0

    def pid(self, set, position, velocity):
        error = set - position
        self.error_int += error
        #velocity = - velocity
        velocity = error - self.error_prev
        self.error_prev = error

        u = self.k_p*error + self.k_d*velocity + self.k_i*self.error_int
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
