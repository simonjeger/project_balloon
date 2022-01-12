import numpy as np
import time
import os
import time
import json
import sys

from utils.raspberrypi_com import raspi_com

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def receive(file_name):
    successful = False
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    t_start = time.time()
    corrupt = False
    while not successful:
        with open(path + file_name) as json_file:
            try:
                data = json.load(json_file)
                successful = True
            except:
                corrupt = True
    if corrupt:
        print('RBP: data corrupted, lag of ' + str(np.round(time.time() - t_start,3)) + '[s]')
    return data

def send(self, data):
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    with open(path + 'action.txt', 'w') as f:
        f.write(json.dumps(data))
    return data

com = raspi_com()
interval = 60 #s

global_start = time.time()
while True:
    t_start = time.time()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(5)
        print('Waiting for the algorithm to publish')

    elif not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/data.txt'):
        time.sleep(5)
        print('Waiting for the low level controller to publish')

    else:
        try:
            action = receive('action.txt')
            data = receive('data.txt')
            info = ''
            info += 'gps_lat: ' + str(data['gps_lat']) + ', '
            info += 'gps_lon: ' + str(data['gps_lon']) + ', '
            info += 'gps_height: ' + str(data['gps_height']) + ', '
            info += 'rel_pos_est: ' + str(data['rel_pos_est']) + ', '
            info += 'U: ' + str(data['U']) + ', '
            info += 'action: ' + str(action['action'])
            info += 'overrite_action: ' + str(action['overrite_action'])
            com.send_sms(info)

            message = com.receive_sms()
            overrite_action = False
            try:
                overrite_action = float(message)
            action['overrite_action'] = overrite_action
            send(action)

        except KeyboardInterrupt:
            print("Maual kill")
            com.power_off()
            sys.exit()

        except:
            print("Something fatal broke down at " + str(int(t_start - global_start)) + ' s after start')
            com.power_off()
            sys.exit()

    wait = interval - (t_start - global_start)
    if wait > 0:
        time.sleep(wait)
