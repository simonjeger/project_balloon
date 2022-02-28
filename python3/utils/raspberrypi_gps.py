import RPi.GPIO as GPIO

import serial
import time
from filelock import FileLock
import numpy as np

import logging
logger=logging.getLogger()

class raspi_gps:
	def __init__(self,path):
		self.ser = serial.Serial('/dev/ttyUSB2',115200)
		self.ser.timeout = 2
		self.ser.flushInput()

		self.path = path
		self.power_key = 6

		try:
			self.power_on()
		except:
			if self.ser != None:
				self.ser.close()
			self.power_off()
			GPIO.cleanup()

	def send_at(self,command,back):
		rec_buff = b''
		self.ser.flush()
		self.ser.write((command+'\r\n').encode())

		start = time.time()
		while time.time() - start < 4:
			rec_buff += self.ser.read_until(b'\r\n')
			if ('OK'.encode() in rec_buff) | ('ERROR'.encode() in rec_buff):
				break

		if rec_buff != '':
			if back.encode() not in rec_buff:
				return 0, rec_buff
			else:
				return 1, rec_buff
		else:
			logger.error('GPS: not ready')
			return 0, rec_buff

	def init_gps_position(self,max_cycles=60):
		answer = 0
		for c in range(max_cycles):
			logger.info('GPS: getting fixture (' + str(c) + ' out of ' + str(max_cycles) + ' tries)')
			pos = self.get_gps_position()
			if pos is not None:
				logger.info('GPS: initial fixture found')
				return pos

	def get_gps_position(self):
		with FileLock(self.path + 'waveshare.lock'):
			answer = 0
			self.send_at('AT+CGPS=1,1','OK')
			self.info()
			answer, result = self.send_at('AT+CGPSINFO','+CGPSINFO: ')
			#self.send_at('AT+CGPS=0','OK') this actually breaks the whole thing
			if 1 == answer:
				answer = 0
				if ',,,,,,' in str(result):
					logger.error('GPS: empty ' + str(result))
					time.sleep(1)
				else:
					result_array = str(result)[30::].split(",")
					lat = self.convert_min_to_dec(result_array[0])
					lat_dir = result_array[1]
					lon = self.convert_min_to_dec(result_array[2])
					lon_dir = result_array[3]
					year = int(result_array[4][0:2])
					month = int(result_array[4][2:4])
					day = int(result_array[4][4:6])
					hour = int(result_array[5][0:2])
					minute = int(result_array[5][2:4])
					second = int(result_array[5][4:6])
					height = float(result_array[6])

					if lat_dir == 'S':
						lat *= -1
					if lon_dir == 'W':
						lon *= -1

					logger.info('GPS: lat ' + str(np.round(lat,6)) + ', lon ' + str(np.round(lon,6)) + ', height ' + str(np.round(height,2)) + ' m')
					return lat,lon,height
			else:
				logger.error('GPS: error ' + str(result))

	def convert_min_to_dec(self,lat_or_lon):
		deg = str(lat_or_lon)[0:-9]
		min = str(lat_or_lon)[-9:-1]
		res = float(deg) + float(min)/60
		return res

	def info(self):
		answer, result = self.send_at('AT+CGPS?', 'CGPS')
		rec_string = str(result).split("\\")
		logger.info('GPS: info ' + str(rec_string))

	def power_on(self):
		with FileLock(self.path + 'waveshare.lock'):
			GPIO.setmode(GPIO.BCM)
			GPIO.setwarnings(False)
			GPIO.setup(self.power_key,GPIO.OUT)
			time.sleep(0.1)
			GPIO.output(self.power_key,GPIO.HIGH)
			time.sleep(2)
			GPIO.output(self.power_key,GPIO.LOW)
			time.sleep(20)
			self.ser.flushInput()
			logger.info('GPS: powered on')

	def power_off(self):
		with FileLock(self.path + 'waveshare.lock'):
			GPIO.output(self.power_key,GPIO.HIGH)
			time.sleep(3)
			GPIO.output(self.power_key,GPIO.LOW)
			time.sleep(18)
			logger.info('GPS: powered off')
