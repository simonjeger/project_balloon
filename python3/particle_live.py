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
        offset_vicon_z = -0.725

        self.position[0] = data.pose.position.x + offset_vicon_x
        self.position[0] = 0 + offset_vicon_x #to turn into 2d problem
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
    while not successful:
        with open(path + 'action.txt') as json_file:
            try:
                data = json.load(json_file)
                successful = True
            except:
                print('data corrupted, will try again')
    return data

def update_est(position,u,c):
    est_x.one_cycle(0,c,position[0])
    est_y.one_cycle(0,c,position[1])
    est_z.one_cycle(u,c,position[2])
    position_est = [est_x.xhat_0[0], est_y.xhat_0[0], est_z.xhat_0[0]]
    return position_est

character = character_vicon()
offset = 0
scale = 0.8

llc = ll_controler()

est_x = ekf(0.1, character.position[0])
est_y = ekf(0.1, character.position[1])
est_z = ekf(0.1, character.position[2])

path = []
path_est = []
not_done = True

U = 0
min_proj_dist = np.inf
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

    else:
        data = receive()
        action = data['action']
        target = data['target']
        ceiling = data['ceiling']

        character.set_vicon()
        terrain = 0
        ceiling = 3

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

        u = offset + llc.pid(action, rel_pos, rel_vel)*(1-offset)*scale
        u = -0.8
        call(u)
        if (not not_done) | (action < 0):
            u = 0
            call(u)
            break

        #c = self.area*self.rho_air*self.c_w/(2*self.mass_total)
        c = 1
        position_est = update_est(character.position,u,c)

        path.append(np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())
        path_est.append(np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())

        stop = time.time()
        delta_t = stop - start
        U += abs(u*delta_t)

        # find min_proj_dist
        render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
        residual = target - np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']])
        min_proj_dist_prop = np.sqrt((residual[0]*render_ratio/yaml_p['radius_xy'])**2 + (residual[1]*render_ratio/yaml_p['radius_xy'])**2 + (residual[2]/yaml_p['radius_z'])**2)
        if min_proj_dist_prop < min_proj_dist:
            min_proj_dist = min_proj_dist_prop

        data = {
        'U': U,
        'position': np.divide(character.position,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'velocity': np.divide(character.velocity,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path': path,
        'position_est': position_est,
        'path_est': path_est,
        'measurement': [est_x.wind(), est_y.wind()],
        'min_proj_dist': min_proj_dist,
        'not_done': not_done}

    send(data)
