import numpy as np

class ekf():
    def __init__(self, delta_t):
        self.delta_t = delta_t

        self.xhat_0 = np.array([0,0,0])     #predicted state
        self.xhat_min1 = np.array([0,0,0])
        self.xhat_min2 = np.array([0,0,0])
        self.P_0 = np.eye((3))              #error covariance (dim_x, dim_x)
        self.W_0 = np.ones((3))             #noise covariance (dim_x, dim_x)
        self.Q_0 = np.eye((3))              #process noise (dim_x, dim_x)
        self.R_0 = [1, 100, 100]                #measurement noise (dim_z, dim_z)
        self.H_0 = np.eye((3))              #measurement function (dim_x, dim_x)

    def f(self):
        p = self.xhat_0[0]
        v = (self.xhat_0[0] - self.xhat_min1[0])/self.delta_t
        a = (v - (self.xhat_min1[0] - self.xhat_min2[0])/self.delta_t)/self.delta_t

        return np.array([p,v,a])

    def h(self):
        return [self.xhat_0[0], 0, 0]


    def project_state(self):
        self.xhat_0 = self.f()
        self.xhat_min2 = self.xhat_min1
        self.xhat_min1 = self.xhat_0

    def set_A_0(self):
        self.A_0 = np.array([[1, self.delta_t, 0],[0, 1, self.delta_t], [0, 0, -np.sign(self.xhat_0[1])*self.c*self.xhat_0[1]]])

    def project_error(self):
        self.set_A_0()
        self.P_0 = self.A_0*self.P_0*self.A_0.T + self.W_0*self.Q_0*self.W_0.T

    def kalman_gain(self):
        self.S_0 = self.H_0*self.P_0*self.H_0 + self.R_0
        self.K_0 = self.P_0*self.H_0*np.linalg.inv(self.S_0)

    def update_estimate(self):
        y_0 = self.z_0 - self.h()
        self.xhat_0 = self.xhat_0 + np.matmul(self.K_0,y_0)

    def update_covariance(self):
        self.P_0 = (np.eye((len(self.xhat_0))) - self.K_0*self.H_0)*self.P_0


    def predict(self, u_0, c):
        self.c = c
        self.u_0 = u_0
        self.project_state()
        self.project_error()

    def correct(self, z_0):
        self.z_0 = np.zeros_like(self.xhat_0)
        self.z_0[0] = z_0
        self.kalman_gain()
        self.update_estimate()
        self.update_covariance()


    def one_cycle(self, u_0, c, z_0):
        self.predict(u_0, c)
        self.correct(z_0)

    def wind(self):
        block = (self.u_0*self.delta_t + self.xhat_min1[1] - self.xhat_0[1])/(self.delta_t*self.c)
        return self.xhat_min1[1] - np.sign(block)*np.sqrt(abs(block))
