import numpy as np
import scipy
from random import gauss

class character():
    def __init__(self, size_x, size_z, start, target, T, wind_compressed):
        self.mass = 1000
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45

        self.size_x = size_x
        self.size_z = size_z
        self.start = start
        self.target = target
        self.t = T
        self.action = 2

        self.position = self.start
        self.residual = self.target - self.position
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_z = self.position[1] - 0
        self.max_z = self.size_z - self.position[1]

        self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z, self.t], wind_compressed.flatten()), axis=0)
        self.path = [self.position.copy(), self.position.copy()]
        self.min_distance = np.sqrt(self.residual[0]**2 + self.residual[1]**2)

    def update(self, action, wind, wind_compressed):
        self.action = action

        in_bounds = self.move_particle(wind, 100)

        # update state
        self.residual = self.target - self.position
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_z = self.position[1] - 0
        self.max_z = self.size_z - self.position[1]
        self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z, self.t], wind_compressed.flatten()), axis=0)

        min_distance = np.sqrt(self.residual[0]**2 + self.residual[1]**2)
        if min_distance < self.min_distance:
            self.min_distance = min_distance

        # reduce flight length by 1 second
        self.t -= 1

        return in_bounds

    def move_particle(self, wind, n):
        c = self.area*self.rho*self.c_w/(2*self.mass)
        b = (self.action - 1)*30 / self.mass #random number so it looks reasonable
        delta_t = 1/n

        p_x = (self.path[-1][0] - self.path[-2][0])/delta_t
        p_z = (self.path[-1][1] - self.path[-2][1])/delta_t
        for _ in range(n):
            coord = [int(i) for i in np.floor(self.position)]
            in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x = wind[coord[0], coord[1]][0]
                w_z = wind[coord[0], coord[1]][1]

                w_x += gauss(0,wind[coord[0], coord[1]][2]/np.sqrt(n)) #is it /sqrt(n) or just /n?
                w_z += gauss(0,wind[coord[0], coord[1]][2]/np.sqrt(n))

                v_x = (np.sign(w_x - p_x) * (w_x - p_x)**2 * c + 0)*delta_t + p_x
                v_z = (np.sign(w_z - p_z) * (w_z - p_z)**2 * c + b)*delta_t + p_z

                # update
                self.position += [v_x*delta_t, v_z*delta_t]
                p_x = v_x
                p_z = v_z

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it
        return in_bounds
