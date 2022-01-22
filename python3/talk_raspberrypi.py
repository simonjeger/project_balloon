import numpy as np
import time
import os
import time
import datetime
import pytz
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

def send(data):
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    with open(path + 'action.txt', 'w') as f:
        f.write(json.dumps(data))
    return data

com = raspi_com(yaml_p['phone_number'], yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/')
interval = 60 #s
action_overwrite = False
stop_logger = False
message_fail = 0

global_start = time.time()
timestamp_start = datetime.datetime.today().astimezone(pytz.timezone("Europe/Zurich"))

message, timestamp = com.receive_last_sms()

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
            info += 'gps_lat: ' + str(np.round(data['gps_lat'],6)) + ', '
            info += 'gps_lon: ' + str(np.round(data['gps_lon'],6)) + ', '
            info += 'gps_height: ' + str(np.round(data['gps_height'],1)) + ', '
            info += 'rel_pos_est: ' + str(np.round(data['rel_pos_est'],3)) + ', '
            info += 'u: ' + str(np.round(data['u'],3)) + ', '
            info += 'action: ' + str(np.round(action['action'],3)) + ', '
            info += 'action_overwrite: ' + str(action['action_overwrite']) + ', '
            info += 'stop_logger: ' + str(action['stop_logger']) + ', '

            #info += 'action_overwrite: ' + str(action['action_overwrite']) + ', '

            try:
                com.send_sms(info)
                message_fail = 0
            except:
                message_fail += 1
                print('Could not send')
            try:
                message, timestamp = com.receive_last_sms()
                if timestamp > timestamp_start:
                    if message = 'stop':
                        stop_logger = True
                    try:
                        action_overwrite = float(message)
                        print('Overwriting action to: ' + str(action_overwrite))
                    except:
                        print('Could not turn into float: ' + message)
                else:
                    print('No new message, the latest one is from ' + str(timestamp))
            except:
                print('Could not receive')

            if message_fail >= 5:
                action_overwrite = -1
                print('Set action_overwrite = -1 because of message_fail')

            action['action_overwrite'] = action_overwrite
            action['stop_logger'] = stop_logger
            send(action)

            wait = interval - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)

        except KeyboardInterrupt:
            print("Maual kill")
            com.power_off()
            sys.exit()

        except:
            print("Something fatal broke down at " + str(int(t_start - global_start)) + ' s after start')
            com.power_off()
            sys.exit()
