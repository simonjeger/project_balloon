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

# Delay start so files don't get overwritten during start up
if yaml_p['environment'] == 'gps':
    time.sleep(160)

import logging
path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger/'
logging.basicConfig(filename=path+'talk_raspberrypi.log', format='%(asctime)s %(message)s', filemode='w')
logging.getLogger().addHandler(logging.StreamHandler())
logger=logging.getLogger()
logger.setLevel(logging.INFO)

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
        logger.warning('RBP: data corrupted, lag of ' + str(np.round(time.time() - t_start,3)) + '[s]')
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
com_fail = 0
thrust_fail = 0
gps_hist = []
gps_fail = 0

global_start = time.time()
timestamp_start = datetime.datetime.today().astimezone(pytz.timezone("Europe/Zurich"))
bool_data = False
bool_action = False
bool_start = False

while True:
    t_start = time.time()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/data.txt'):
        time.sleep(5)
        logger.info('Waiting for the low level controller to publish')
    else:
        if bool_data == False:
            com.send_sms('Low level controller ready')
        time.sleep(5)
        bool_data = True

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(5)
        logger.info('Waiting for the algorithm to publish')
    else:
        if bool_action == False:
            com.send_sms('Algorithm ready')
        time.sleep(5)
        bool_action = True

    if (bool_data & bool_action):
        try:
            action = receive('action.txt')
            data = receive('data.txt')

            try:
                message, timestamp = com.receive_last_sms()
                if timestamp > timestamp_start:
                    if message == 'stop':
                        stop_logger = True
                        logger.info('Stopping logger')
                    elif message == 'start':
                        bool_start = True
                        logger.info('Starting algorithm')
                    else:
                        try:
                            action_overwrite = float(message)
                            logger.info('Overwriting action to: ' + str(action_overwrite))
                        except:
                            logger.error('Could not turn into float: ' + message)
                else:
                    logger.info('No new message, the latest one is from ' + str(timestamp) + ' and reads: ' + message)
            except:
                logger.error('Could not receive')

            info = ''
            info += 'action: ' + str(np.round(action['action'],3)) + ', '
            info += 'action_asl: ' + str(np.round(action['action_asl'],1)) + 'm, '
            info += 'action_ow: ' + str(action['action_overwrite']) + ', '
            info += 'lat: ' + str(np.round(data['gps_lat'],6)) + ', '
            info += 'lon: ' + str(np.round(data['gps_lon'],6)) + ', '
            info += 'height: ' + str(np.round(data['gps_height'],1)) + 'm, '
            info += 'rel_pos: ' + str(np.round(data['rel_pos_est'],3)) + ', '
            info += 'u: ' + str(np.round(data['u'],1)) + ', '
            info += 'stop_logger: ' + str(action['stop_logger'])

            # com fail
            if com_fail >= 5:
                action_overwrite = -1
                info += ', com_error'
                logger.error('com_fail, set action_overwrite = -1')

            # thrust fail
            if (data['velocity'][2] < 0) & (data['u'] > 0):
                thrust_fail += 1
            else:
                thrust_fail = 0

            if thrust_fail >= 3:
                info += ', thrust_warning'
                logger.error('thrust_fail')

            # gps fail
            gps_hist.append([data['gps_lat'], data['gps_lon']])
            if len(gps_hist) > 2:
                if (gps_hist[-1][0] - gps_hist[-2][0])**2 + (gps_hist[-1][1] - gps_hist[-2][1])**2 < 1e-8:
                    gps_fail += 1
                else:
                    gps_fail = 0

            if gps_fail >= 5:
                info += ', gps_warning'
                logger.error('gps_fail')

            try:
                com.send_sms(info)
                com_fail = 0
            except:
                com_fail += 1
                logger.error('Could not send')

            action['action_overwrite'] = action_overwrite
            action['stop_logger'] = stop_logger
            send(action)

            # send starting signal to algorithm if ready
            if bool_start:
                with open(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/start.txt', 'w') as f:
                    f.write('start')

            wait = interval - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)

        except KeyboardInterrupt:
            logger.info("Maual kill")
            com.power_off()
            sys.exit()

        except:
            logger.error("Something fatal broke down at " + str(int(t_start - global_start)) + ' s after start')
            com.power_off()
            sys.exit()
