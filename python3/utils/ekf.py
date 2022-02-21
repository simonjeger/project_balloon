import numpy as np
import matplotlib.pyplot as plt #for debugging only

class ekf():
    def __init__(self, x_0):
        self.xhat_0 = np.array([x_0,0,0,0])                                 #predicted state

        self.z_hist = [x_0,x_0]                                                 #memory states

        self.P_0 = np.ones((4))                                             #error covariance (dim_x, dim_x)
        self.W_0 = np.ones((4))                                             #noise covariance (dim_x, dim_x)
        self.Q_0 = np.eye((4))                                              #process noise (dim_x, dim_x)
        self.Q_0[0,0] = 1
        self.Q_0[1,1] = 1
        self.Q_0[2,2] = 1                                                   #the acceleration is not actually zero
        self.Q_0[3,3] = 1                                                   #the wind is not actually zero
        self.R_0 = np.eye((4))                                              #measurement noise (dim_z, dim_z)
        self.R_0[0,0] = 20
        self.R_0[1,1] = 40
        self.R_0[2,2] = 100
        self.R_0[3,3] = 10
        self.H_0 = np.eye((4))                                              #measurement function (dim_x, dim_x)

        self.hist_p = []                        #for plotting
        self.hist_v = []
        self.hist_a = []
        self.meas_p = []
        self.pred = []
        self.pred_basic = []

    def f(self):
        #p = self.xhat_0[0] + self.xhat_0[1]*self.delta_t
        #v = self.xhat_0[1] + self.xhat_0[2]*self.delta_t

        #v_rel = v - self.xhat_0[1]
        #a = self.u_0 - np.sign(v_rel)*self.c*v_rel**2
        a = 0 #because acceleration is so quick anyway
        v = self.xhat_0[1] + a
        p = self.xhat_0[0] + v*self.delta_t
        o = self.xhat_0[3]
        return np.array([p,v,a,o])

    def h(self):
        return self.xhat_0

    def project_state(self):
        self.xhat_0 = self.f()

    def set_A_0(self):
        if self.xhat_0[2] + self.u_0 == 0:
            self.A_0 = np.array([[1, self.delta_t, 0, 0],[0, 1, self.delta_t, 0], [0, -2*self.c*self.xhat_0[1], 0, 0], [0, 1, 0, 0]])
        else:
            self.A_0 = np.array([[1, self.delta_t, 0, 0],[0, 1, self.delta_t, 0], [0, -2*self.c*self.xhat_0[1], 0, 0], [0, 1, -self.delta_t - 1/(2*self.c*np.sign(self.xhat_0[2] + self.u_0)*np.sqrt((abs(self.xhat_0[2] + self.u_0))/self.c)),0]])

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

    def measure(self, z_0):
        self.z_hist.append(z_0)
        p = self.z_hist[-1]
        v = (self.z_hist[-1] - self.z_hist[-2])/self.delta_t
        a = ((self.z_hist[-1] - self.z_hist[-2])/self.delta_t - (self.z_hist[-2] - self.z_hist[-3])/self.delta_t)/self.delta_t

        v_0 = (self.z_hist[-1] - self.z_hist[-2])/self.delta_t
        v_min1 = (self.z_hist[-2] - self.z_hist[-3])/self.delta_t
        block = (self.u_0*self.delta_t + v_min1 - v_0)/(self.delta_t*self.c)
        o =  v_min1 - np.sign(block)*np.sqrt(abs(block))

        self.z_0 = [p, v, a, o]
        #print('measure: ' + str(self.z_0))

    def predict(self, u_0, c):
        self.c = c
        self.u_0 = u_0
        self.project_state()
        self.project_error()
        #print('predict: ' + str(self.xhat_0))

    def correct(self, z_0):
        self.measure(z_0)
        self.kalman_gain()
        self.update_estimate()
        self.update_covariance()
        #print('correct: ' + str(self.xhat_0))

    def one_cycle(self, u_0, z_0, c, delta_t, measurement=True):
        self.delta_t = delta_t
        self.predict(u_0, c)
        if measurement:
            self.correct(z_0)

        self.hist_p.append(self.xhat_0[0])
        self.hist_v.append(self.xhat_0[1])
        self.hist_a.append(self.xhat_0[2])
        self.meas_p.append(self.z_0[0])
        self.pred.append(self.wind())

        if len(self.hist_v) > 1:
            block = (u_0*self.delta_t + self.hist_v[-2] - self.hist_v[-1])/(self.delta_t*c)
            self.pred_basic.append(self.hist_v[-2] - np.sign(block)*np.sqrt(abs(block)))

    def wind(self):
        return self.xhat_0[3]

    def plot(self):
        fig, axs = plt.subplots(4)
        axs[0].plot(self.meas_p, color='red')
        axs[0].plot(self.hist_p, color='green')
        axs[0].set_ylabel('position')

        axs[1].plot(np.arange(1,len(self.hist_a),1),np.diff(self.meas_p)/self.delta_t, color='red')
        axs[1].plot(self.hist_v, color='green')
        axs[1].scatter([0], [0], color='white')
        axs[1].set_ylabel('velocity')

        axs[2].plot(np.arange(2,len(self.hist_a),1),np.diff(np.diff(self.meas_p))/self.delta_t**2, color='red')
        axs[2].plot(self.hist_a, color='green')
        axs[2].scatter([0], [0], color='white')
        axs[2].set_ylabel('acceleration')

        axs[3].plot(np.arange(1,len(self.hist_a),1),self.pred_basic, color='red')
        axs[3].plot(self.pred, color='green')
        axs[3].scatter([0], [0], color='white')
        axs[3].set_ylabel('offset')

        plt.savefig('debug_ekf.png')
        plt.close()

        """
        self.hist_p = []
        self.hist_v = []
        self.hist_a = []
        self.meas_p = []
        self.pred = []
        self.pred_basic = []
        """
