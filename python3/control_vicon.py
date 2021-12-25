import rospy
from geometry_msgs.msg import PoseStamped, Point
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
import numpy as np
import time
import urllib.request
import sys
import select
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import os
import time
import json

from build_ll_controller import ll_controler
from utils.ekf import ekf

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class character_vicon():
    def __init__(self):
        self.position = [0,0,0]
        self.velocity = [0,0,0]

        self.set_vicon()
        #print(self.position)

    def set_vicon(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/vrpn_client_node/balloon/pose", PoseStamped, self.callback_pos)
        rospy.Subscriber("/vrpn_client_node/balloon/twist", TwistStamped, self.callback_vel)

        r = rospy.Rate(10)
        r.sleep()

    def callback_pos(self, data):
        offset_vicon_x = 1.25
        offset_vicon_y = 0.75
        offset_vicon_z = -0.725 #+0.28


        self.position[0] = data.pose.position.x + offset_vicon_x
        #self.position[0] = 0 + offset_vicon_x #to turn into 2d problem
        self.position[1] = data.pose.position.y + offset_vicon_y
        self.position[2] = data.pose.position.z + offset_vicon_z

    def callback_vel(self, data):
        self.velocity[0] = data.twist.linear.x
        self.velocity[1] = data.twist.linear.y
        self.velocity[2] = data.twist.linear.z

def call(con):
    url = "http://192.168.0.238/"  + str(con)
    n = urllib.request.urlopen(url).read() # get the raw html data in bytes (sends request and warn our esp8266)
    n = n.decode("utf-8") # convert raw html bytes format to string :3

    # data = n
    data = n.split() 			#<optional> split datas we got. (if you programmed it to send more than one value) It splits them into seperate list elements.
    #data = list(map(int, data)) #<optional> turn datas to integers, now all list elements are integers.
    return data

def send(data):
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    with open(path + 'data.txt', 'w') as f:
        f.write(json.dumps(data))
    return data

def receive():
    successful = False
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    start = time.time()
    corrupt = False
    while not successful:
        with open(path + 'action.txt') as json_file:
            try:
                data = json.load(json_file)
                successful = True
            except:
                corrupt = True
    if corrupt:
        print('data corrupted, lag of ' + str(np.round(time.time() - start,3)) + '[s]')
    return data

def update_est(position,u,c,delta_t,delta_f_up,delta_f_down,mass_total):
    force_est = (max(0,u)*delta_f_up + min(0,u)*delta_f_down)/yaml_p['unit_z']/mass_total
    est_x.one_cycle(0,position[0],c,delta_t)
    est_y.one_cycle(0,position[1],c,delta_t)
    est_z.one_cycle(force_est,position[2],c,delta_t)
    position_est = [est_x.xhat_0[0], est_y.xhat_0[0], est_z.xhat_0[0]]
    return position_est

def plot_pid(plot_time, plot_pos_z, plot_vel, plot_u, plot_action):
    plt.plot(plot_time, plot_pos_z)
    plt.plot(plot_time, plot_vel)
    plt.plot(plot_time, plot_u)
    plt.plot(plot_time, plot_action)
    plt.xlabel('[s]')
    plt.ylabel('[m]')
    plt.title('max velocity: ' + str(np.round(np.max(plot_vel),2)) + 'm/s')
    plt.savefig('debug_pid.png')
    plt.close()

def plot(plot_pos_x, plot_pos_y, plot_pos_z, target, plot_residual):
    # loading file
    filepath = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/3dplot.obj'
    file = open(filepath, 'wb')
    if os.path.isfile(filepath):
        ax = pickle.load(filepath)
    else:
        ax = plt.axes(projection='3d')

    # plotting
    ax.plot3D(plot_pos_x, plot_pos_y, plot_pos_z)
    ax.plot3D(np.linspace(0,yaml_p['size_x']*yaml_p['unit_xy'], 10), [target[1]*yaml_p['unit_xy']]*10, [target[2]*yaml_p['unit_z']]*10)

    # mark the border of the box
    ax.set_xlim3d(0, yaml_p['size_x']*yaml_p['unit_xy'])
    ax.set_ylim3d(0, yaml_p['size_y']*yaml_p['unit_xy'])
    ax.set_zlim3d(0, yaml_p['size_z']*yaml_p['unit_z'])

    ax.set_title('min 2d residual: ' + str(np.round(np.linalg.norm(plot_residual),3)) + ' m')

    file = open(filepath, 'wb')
    pickle.dump(ax,file)
    plt.show()
    plt.close()

def tuning(delta_t):
    if delta_t < 2:
        action = 0
    elif delta_t < 22:
        action = 0.8
    elif delta_t < 40:
        action = 0.2
    else:
        action = -1
    return action

character = character_vicon()
offset = 0.11
scale = 0.7

llc = ll_controler()

est_x = ekf(character.position[0]/yaml_p['unit_xy'])
est_y = ekf(character.position[1]/yaml_p['unit_xy'])
est_z = ekf(character.position[2]/yaml_p['unit_z'])

path = []
path_est = []
not_done = True

plot_time = []
plot_pos_x = []
plot_pos_y = []
plot_pos_z = []
plot_vel = []
plot_u = []
plot_action = []
plot_residual = np.inf

U = 0
min_proj_dist = np.inf

global_start = time.time()
while True:
    start = time.time()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(1)
        print('waiting for the algorithm to publish')

        data = {
        'U': U,
        'position': np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'velocity': np.divide(character.velocity,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path': [],
        'position_est': np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path_est': [],
        'measurement': [0, 0],
        'min_proj_dist': 0,
        'not_done': not_done}

        action = 0 #there is no command to reveive anything from, so I assume the action = 0 (only important for break out condition at the end)

    else:
        data = receive()
        action = data['action']
        target = data['target']
        ceiling = data['ceiling']*yaml_p['unit_z'] #in meters
        c = data['c']
        delta_f_up = data['delta_f_up']
        delta_f_down = data['delta_f_down']
        mass_total = data['mass_total']

        character.set_vicon()
        terrain = 0

        rel_pos = (character.position[2] - terrain)/(ceiling-terrain)
        rel_vel = character.velocity[2] / (ceiling-terrain)

        # check if done or not
        if (character.position[0] < 0) | (character.position[0]/yaml_p['unit_xy'] > yaml_p['size_x'] - 1):
            print('x out of bounds')
            not_done = False
        if (character.position[1] < 0) | (character.position[1]/yaml_p['unit_xy'] > yaml_p['size_y'] - 1):
            print('y out of bounds')
            not_done = False
        #if (rel_pos < 0) | (rel_pos >= 1):
        #    print('z out of bounds')
        #    not_done = False
        #if self.t < 0: #check if flight time is over
        #    not_done = False
        #if self.battery_level < 0: #check if battery is empty
        #    not_done = False

        #action = tuning(time.time() - global_start)
        u_raw = llc.pid(action, rel_pos, rel_vel)
        u = offset + u_raw*scale

        if yaml_p['mode'] == 'tuning':
            print(u)

        call(u)
        if (not not_done) | (action < 0):
            u = 0
            call(u)
            plot_pid(plot_time, plot_pos_z, plot_vel, plot_u, plot_action)
            #plot(plot_pos_x, plot_pos_y, plot_pos_z, target, plot_residual)

        stop = time.time()
        delta_t = stop - start

        position_est = update_est(np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]),u,c,delta_t,delta_f_up,delta_f_down,mass_total))

        path.append(np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())
        path_est.append(np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())

        plot_pos_x.append(character.position[0])
        plot_pos_y.append(character.position[1])
        plot_pos_z.append(character.position[2])
        plot_vel.append(character.velocity[2])
        plot_u.append(u)
        plot_action.append(action*ceiling)
        plot_time.append(time.time())

        U += abs(u*delta_t)

        # find min_proj_dist
        render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
        residual = target - np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']])
        min_proj_dist_prop = np.sqrt((residual[1]*render_ratio/yaml_p['radius_xy'])**2 + (residual[2]/yaml_p['radius_z'])**2) #only 2d case!
        min_dist_prop = np.sqrt((residual[1]*render_ratio)**2 + (residual[2])**2)*yaml_p['unit_z']
        if min_proj_dist_prop < min_proj_dist:
            min_proj_dist = min_proj_dist_prop
            min_dist = min_dist_prop
            plot_residual = target*np.array([yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]) - character.position

        data = {
        'U': U,
        'position': np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'velocity': np.divide(character.velocity,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path': path,
        'position_est': position_est.tolist(),
        'path_est': path_est,
        'measurement': [est_x.wind(), est_y.wind()],
        'min_proj_dist': min_proj_dist,
        'min_dist': min_dist,
        'not_done': not_done}

    send(data)
    if (not not_done) | (action < 0):
        break
