import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# measurements before transformation function is used
energy = np.array([0.078,0.112,0.276,0.472,0.918,1.310,2.014,2.756,3.952,4.926,6.544,7.762]) #A
energy *= 16 #V
throttle = np.array([0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,0.375])
thrust = np.array([0.000,0.000,0.020,0.040,0.078,0.106,0.152,0.190,0.248,0.286,0.342,0.374,0.432,0.466,0.490,0.500]) #kg
thrust *= 9.81 #m/s^2

# measurements after transformation function is used
throttle2 = np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00])
thrust2 = np.array([0.000,0.030,0.052,0.076,0.104,0.118,0.150,0.168,0.186,0.222,0.240,0.260,0.296,0.312,0.328,0.364,0.382,0.418,0.452,0.480,0.496]) #kg
thrust2 *= 9.81 #m/s^2

def is_func(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def should_func(x):
    return 5*x

def trans(x,p):
    s = np.sign(x)
    x = abs(x)
    a,b,c,d,e,f,g = p
    return s*abs(a + b*x**c + d*x**e + f*x**g)

def error(p):
    x = np.linspace(0,1,1000)
    return np.sum(abs(should_func(x) - is_func(trans(x,p), *popt)))

fix,axs = plt.subplots(4)
axs[0].plot(throttle[0:len(energy)],energy)
axs[0].set_ylabel('energy [W]')
axs[1].plot(throttle,thrust)
axs[1].set_ylabel('thrust [N]')

popt, pcov = curve_fit(is_func, throttle, thrust)
throttle_fine = np.linspace(0,throttle[-1],1000)
axs[1].plot(throttle_fine, is_func(throttle_fine, *popt),zorder=0)
axs[1].legend(['data',  str(np.round(popt[0],3)) + str(np.round(popt[1],3)) + '*x + ' + str(np.round(popt[2],3)) + '*x^2 + ' + str(np.round(popt[3],3)) + '*x^3'])

res = minimize(error, [0,0,1,0,2,0,3])
throttle_uniform = np.linspace(-1,1,1000)

axs[2].plot(throttle_uniform, trans(throttle_uniform,res.x))
axs[2].set_ylabel('new throttle')
axs[2].legend(['sign(x)*|' + str(np.round(res.x[0],3)) + str(np.round(res.x[1],3)) + '*x^' + str(np.round(res.x[2],3)) + ' + '  + str(np.round(res.x[3],3)) + '*x^'  + str(np.round(res.x[4],3)) + '*x^' + str(np.round(res.x[5],3)) + '|'])

axs[3].plot(throttle_uniform, should_func(throttle_uniform))
axs[3].plot(throttle_uniform[int(len(throttle_uniform)/2):-1], is_func(trans(throttle_uniform[int(len(throttle_uniform)/2):-1],res.x), *popt))
axs[3].plot(throttle2, thrust2)
axs[3].legend(['ideal mapping', 'simulation','measured'])
axs[3].set_xlabel('throttle')
axs[3].set_ylabel('thrust [N]')
print('the constants for the transformation function sign(x)*abs(a + b*x**c + d*x**e + f*x**g): ' + str(res.x))

plt.show()
plt.close()
