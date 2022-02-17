import RPi.GPIO as GPIO

import serial
import time
from filelock import FileLock

import logging
logger=logging.getLogger()

class raspi_gps:
	def __init__(self,path):
		self.ser = serial.Serial('/dev/ttyUSB2',115200)
		self.ser.flushInput()

		self.path = path
		self.power_key = 6
		rec_buff = ''

		try:
			self.power_on()
		except:
			if self.ser != None:
				self.ser.close()
			self.power_off()
			GPIO.cleanup()

	def send_at(self,command,back,timeout):
		with FileLock(self.path + 'waveshare.lock'):
			rec_buff = ''
			self.ser.write((command+'\r\n').encode())
			time.sleep(timeout)
			if self.ser.inWaiting():
				time.sleep(0.01 )
				rec_buff = self.ser.read(self.ser.inWaiting())
			if rec_buff != '':
				if back not in rec_buff.decode():
					return 0, rec_buff
				else:
					return 1, rec_buff
			else:
				logger.error('GPS: not ready')
				return 0, rec_buff

	def init_gps_position(self,max_cycles=120):
		answer = 0
		rec_buff = ''
		for c in range(max_cycles):
			self.send_at('AT+CGPS=0','OK',2)
			self.send_at('AT+CGPS=1,1','OK',2)
			logger.info('GPS: getting fixture (' + str(c) + ' out of ' + str(max_cycles) + ' tries)')
			pos = self.get_gps_position()
			if pos is not None:
				return pos

	def get_gps_position(self):
		answer = 0
		rec_buff = ''
		answer, result = self.send_at('AT+CGPSINFO','+CGPSINFO: ',0.5)
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
				return lat,lon,height
		else:
			logger.error('GPS: error ' + str(result))
			rec_buff = ''
			self.send_at('AT+CGPS=0','OK',2)
			self.send_at('AT+CGPS=1,1','OK',2)

	def convert_min_to_dec(self,lat_or_lon):
		deg = str(lat_or_lon)[0:-9]
		min = str(lat_or_lon)[-9:-1]
		res = float(deg) + float(min)/60
		return res

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
