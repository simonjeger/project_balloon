import time
import board
import adafruit_bmp3xx

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
		self.bmp.sea_level_pressure = yaml_p['QNH']

	def get_altitude(self):
		#print('alt: ' + str(self.bmp.altitude))
		return self.bmp.altitude
