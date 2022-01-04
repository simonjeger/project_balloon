import time
import board
import adafruit_bmp3xx
import numpy as np
from scipy.ndimage import gaussian_filter

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class raspi_alt:
    def __init__(self):
        i2c = board.I2C()
        self.bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c, address=0x77)

    def set_QNH(self,terrain):
        QNH_min = 930
        QNH_max = 1070
        x = np.arange(QNH_min,QNH_max,0.1) #range of realistic QNH values at a resolution of 0.1
        y = []
        for x_i in x:
            y.append(self.error(x_i,terrain))
        y = gaussian_filter(y,sigma=10)
        QNH = x[np.argmin(y)]
        self.bmp.sea_level_pressure = QNH

        range = (QNH_max-QNH_min)*0.1
        if (QNH < QNH_min + range) | (QNH > QNH_max - range):
            print('WARNING: Choose larger QNH-range')
        print('ALT: QNH set at ' + str(np.round(QNH,1)) + ' hPa')

    def error(self,QNH,terrain):
        self.bmp.sea_level_pressure = QNH
        return abs(terrain*yaml_p['unit_z'] - self.bmp.altitude)

    def get_altitude(self):
        #print('alt: ' + str(self.bmp.altitude))
        return self.bmp.altitude
