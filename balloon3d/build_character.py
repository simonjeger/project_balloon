import numpy as np
import scipy
from random import gauss

class character():
    def __init__(self, size_x, size_y, size_z, start, target, T, wind_map, wind_compressed):
        self.mass = 1000
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start
        self.target = target
        self.t = T
        self.action = 2

        self.wind_map = wind_map

        self.position = self.start
        self.residual = self.target - self.position
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_y = self.position[1] - 0
        self.max_y = self.size_y - self.position[1]
        self.min_z = self.position[2] - 0
        self.max_z = self.size_z - self.position[2]

        self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], wind_compressed.flatten()), axis=0)
        self.path = [self.position.copy(), self.position.copy()]

    def update(self, action, wind_map, wind_compressed):
        self.action = action
        self.wind_map = wind_map

        in_bounds = self.move_particle(100)

        # update state
        self.residual = self.target - self.position
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_y = self.position[1] - 0
        self.max_y = self.size_y - self.position[1]
        self.min_z = self.position[2] - 0
        self.max_z = self.size_z - self.position[2]
        self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], wind_compressed.flatten()), axis=0)

        # reduce flight length by 1 second
        self.t -= 1

        return in_bounds

    def move_particle(self, n):
        c = self.area*self.rho*self.c_w/(2*self.mass)
        b = (self.action - 1)*30 / self.mass #random number so it looks reasonable
        delta_t = 1/n

        u_prev = (self.path[-1][0] - self.path[-2][0])/delta_t
        v_prev = (self.path[-1][1] - self.path[-2][1])/delta_t
        w_prev = (self.path[-1][2] - self.path[-2][2])/delta_t
        for _ in range(n):
            coord = [int(i) for i in np.floor(self.position)]
            in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_y) & (0 <= coord[2] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                u_pos = self.wind_map[coord[0], coord[1], coord[2]][0]
                v_pos = self.wind_map[coord[0], coord[1], coord[2]][1]
                w_pos = self.wind_map[coord[0], coord[1], coord[2]][2]

                u_pos += gauss(0,self.wind_map[coord[0], coord[1], coord[2]][3]/np.sqrt(n)) #is it /sqrt(n) or just /n?
                v_pos += gauss(0,self.wind_map[coord[0], coord[1], coord[2]][3]/np.sqrt(n))
                w_pos += gauss(0,self.wind_map[coord[0], coord[1], coord[2]][3]/np.sqrt(n))

                u_pres = (np.sign(u_pos - u_prev) * (u_pos - u_prev)**2 * c + 0)*delta_t + u_prev
                v_pres = (np.sign(v_pos - v_prev) * (v_pos - v_prev)**2 * c + 0)*delta_t + v_prev
                w_pres = (np.sign(w_pos - w_prev) * (w_pos - w_prev)**2 * c + b)*delta_t + w_prev

                # update
                self.position += [u_pres*delta_t, v_pres*delta_t, w_pres*delta_t]
                u_prev = u_pres
                v_prev = v_pres
                w_prev = w_pres

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it
        return in_bounds
