import numpy as np
import time
import os
import time
import datetime
import pytz
import json
import sys
import shutil

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
    time.sleep(120)

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

# clear all previous communication files
path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication'
if os.path.exists(path):
    shutil.rmtree(path)
    os.makedirs(path)

com = raspi_com(yaml_p['phone_number'], yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/')
interval_initial = 60 #s
interval = interval_initial
action_overwrite = False
u_overwrite = False
stop_logger = False
com_fail = 0
emergency_landing = False
thrust_fail = 0
gps_hist = []
gps_fail = 0

global_start = time.time()
timestamp_start = datetime.datetime.today().astimezone(pytz.timezone("Europe/Zurich"))
bool_data = False
bool_action = False
bool_start = False
bool_onlyreceive = False
offset = yaml_p['offset']

while True:
    t_start = time.time()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/data.txt'):
        time.sleep(1)
    else:
        if bool_data == False:
            logger.info('Low level controller ready')
            com.send_sms('Low level controller ready')
        time.sleep(1)
        bool_data = True

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(1)
    else:
        if bool_action == False:
            logger.info('Algorithm ready')
            com.send_sms('Algorithm ready')
        time.sleep(1)
        bool_action = True

    if (bool_data & bool_action):
        try:
            action = receive('action.txt')
            data = receive('data.txt')

            info = ''
            if yaml_p['balloon'] == 'outdoor_balloon':
                info += 'act: ' + str(np.round(action['action'],3)) + ', '
            else:
                info += 'act_asl: ' + str(np.round(action['action_asl'],1)) + 'm, '
            info += 'act_ow: ' + str(action['action_overwrite']) + ', '
            info += 'u_ow: ' + str(action['u_overwrite']) + ', '
            info += 'http://maps.google.com/?q=' + str(np.round(data['gps_lat'],6)) + ',' + str(np.round(data['gps_lon'],6))+ ' '
            info += 'hei: ' + str(np.round(data['gps_height'],1)) + 'm, '
            info += 'rel_pos: ' + str(np.round(data['rel_pos_est'],3)) + ', '
            info += 'u: ' + str(np.round(data['u'],1)) + ', '
            info += 'stop_log: ' + str(action['stop_logger'])

            # com fail
            if com_fail >= 10:
                com.min_signal = 0 #so sending out a message becomes more likely
                info += ', com_err'
                logger.error('com_fail, removing communication threshold')

            else:
                com.min_signal = 16 #setting threshold back to the original value

            if (com_fail >= 15) | emergency_landing:
                action_overwrite = -1
                emergency_landing = True
                logger.error('Emergency landing, overwriting action to: ' + str(action_overwrite))

            # thrust fail
            if (data['velocity_est'][2] < 0) & (data['u'] > 0):
                thrust_fail += 1
            else:
                thrust_fail = 0

            if thrust_fail >= 5:
                info += ', thrust_warn'
                logger.error('thrust_fail')

            # gps fail
            gps_hist.append([data['gps_lat'], data['gps_lon'], data['gps_height']])
            if len(gps_hist) > 2:
                if np.sqrt((gps_hist[-1][0] - gps_hist[-2][0])**2 + (gps_hist[-1][1] - gps_hist[-2][1])**2 + (gps_hist[-1][2] - gps_hist[-2][2])**2) < 1e-10:
                    gps_fail += 1
                else:
                    gps_fail = 0

            if gps_fail >= 1:
                info += ', gps_warn'
                logger.error('gps_fail')

            #receive
            try:
                message, timestamp = com.update(info)
                com_fail = 0
                if timestamp > timestamp_start:
                    if (message == 'stop') & (stop_logger == False):
                        stop_logger = True
                        logger.info('Stopping logger')
                    elif (message == 'start') & (bool_start == False):
                        bool_start = True
                        logger.info('Starting algorithm')
                    elif (message == 'tune') & (bool_onlyreceive == False):
                        bool_onlyreceive = True
                        interval = 0
                        com.send_sms('Tuning mode')
                        logger.info('Tuning mode')
                    elif (message == 'fly') & (bool_onlyreceive == True):
                        bool_onlyreceive = False
                        interval = interval_initial
                        com.send_sms('Flying mode')
                        logger.info('Flying mode')
                    elif message == 'u=false':
                        u_overwrite = False
                        logger.info('Setting u_overwrite to False')
                    elif message[0:2] == 'u=':
                        try:
                            u_overwrite = float(message[2::])
                            logger.info('Overwriting u to: ' + str(u_overwrite))
                        except:
                            logger.error('Could not turn into float: ' + message)
                    elif message[0:7] == 'offset=':
                        try:
                            offset = float(message[7::])
                            com.send_sms('Setting offset to: ' + str(offset))
                            logger.info('Setting offset to: ' + str(offset))
                        except:
                            logger.error('Could not turn into float: ' + message)
                    else:
                        try:
                            action_overwrite = float(message)
                            logger.info('Overwriting action to: ' + str(action_overwrite))
                        except:
                            logger.error('Could not turn into float: ' + message)
                else:
                    logger.info('No new message, the latest one is from ' + str(timestamp) + ' and reads: ' + message)
            except:
                logger.error('Could not send and receive')
                com_fail += 1

            action['action_overwrite'] = action_overwrite
            action['u_overwrite'] = u_overwrite
            action['stop_logger'] = stop_logger
            if offset:
                action['offset'] = offset
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
