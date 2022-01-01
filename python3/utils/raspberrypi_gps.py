import RPi.GPIO as GPIO

import serial
import time

class raspi_gps:
	def __init__(self):
		self.ser = serial.Serial('/dev/ttyUSB2',115200)
		self.ser.flushInput()

		power_key = 6
		rec_buff = ''
		rec_buff2 = ''
		time_count = 0

		try:
			self.power_on(power_key)
		except:
			if self.ser != None:
				self.ser.close()
			self.power_down(power_key)
			GPIO.cleanup()

	def send_at(self,command,back,timeout):
		rec_buff = ''
		self.ser.write((command+'\r\n').encode())
		time.sleep(timeout)
		if self.ser.inWaiting():
			time.sleep(0.01 )
			rec_buff = self.ser.read(self.ser.inWaiting())
		if rec_buff != '':
			if back not in rec_buff.decode():
				#print(command + ' ERROR')
				#print(command + ' back:\t' + rec_buff.decode())
				return 0, rec_buff
			else:
				#print(rec_buff.decode())
				return 1, rec_buff
		else:
			print('GPS is not ready')
			return 0, rec_buff

	def get_gps_position(self):
		answer = 0
		#print('Start GPS session...')
		rec_buff = ''
		self.send_at('AT+CGPS=1,1','OK',1)
		time.sleep(2)
		max_cycles = 10
		for c in range(max_cycles):
			answer, result = self.send_at('AT+CGPSINFO','+CGPSINFO: ',1)
			if 1 == answer:
				answer = 0
				if ',,,,,,' in str(result):
					print('GPS is not ready (' + str(c) + ' out of ' + str(max_cycles) + ' tries)')
					time.sleep(1)
				else:
					result_array = str(result)[30::].split(",")
					lat = self.convert_min_to_dec(result_array[0])
					lon = self.convert_min_to_dec(result_array[2])
					year = int(result_array[4][0:2])
					month = int(result_array[4][2:4])
					day = int(result_array[4][4:6])
					hour = int(result_array[5][0:2])
					minute = int(result_array[5][2:4])
					second = int(result_array[5][4:6])
					height = float(result_array[6])
					#print('... lat: ' + str(lat) + ', lon: ' + str(lon) + ', height: ' + str(height))

					return lat,lon,height
			else:
				print('error %d'%answer + ' (' + str(c) + ' out of ' + str(max_cycles) + ' tries)')
				rec_buff = ''
				self.send_at('AT+CGPS=0','OK',1)
			time.sleep(1.5)
		print('ERROR: time out')

	def convert_min_to_dec(self,lat_or_lon):
		deg = str(lat_or_lon)[0:-9]
		min = str(lat_or_lon)[-9:-1]
		res = float(deg) + float(min)/60
		return res

	def power_on(self,power_key):
		print('SIM7600X GPS is starting')
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(power_key,GPIO.OUT)
		time.sleep(0.1)
		GPIO.output(power_key,GPIO.HIGH)
		time.sleep(2)
		GPIO.output(power_key,GPIO.LOW)
		time.sleep(20)
		self.ser.flushInput()
		print('SIM7600X GPS is loging on')

	def power_down(self,power_key):
		print('SIM7600X GPS is loging off')
		GPIO.output(power_key,GPIO.HIGH)
		time.sleep(3)
		GPIO.output(power_key,GPIO.LOW)
		time.sleep(18)
		print('Good bye')
