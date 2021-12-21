import RPi.GPIO as GPIO

import serial
import time

class raspi_gps:
	def __init__(self):
		ser = serial.Serial('/dev/ttyUSB2',115200)
		ser.flushInput()

		power_key = 6
		rec_buff = ''
		rec_buff2 = ''
		time_count = 0

		try:
			self.power_on(power_key)
		except:
			if ser != None:
				ser.close()
			self.power_down(power_key)
			GPIO.cleanup()
		if ser != None:
				ser.close()
				GPIO.cleanup()

	def send_at(self,command,back,timeout):
		rec_buff = ''
		ser.write((command+'\r\n').encode())
		time.sleep(timeout)
		if ser.inWaiting():
			time.sleep(0.01 )
			rec_buff = ser.read(ser.inWaiting())
		if rec_buff != '':
			if back not in rec_buff.decode():
				print(command + ' ERROR')
				print(command + ' back:\t' + rec_buff.decode())
				return 0, rec_buff
			else:
				print(rec_buff.decode())
				return 1, rec_buff
		else:
			print('GPS is not ready')
			return 0, rec_buff

	def get_gps_position(self):
		rec_null = True
		answer = 0
		print('Start GPS session...')
		rec_buff = ''
		self.send_at('AT+CGPS=1,1','OK',1)
		time.sleep(2)
		while rec_null:
			answer, result = self.send_at('AT+CGPSINFO','+CGPSINFO: ',1)
			if 1 == answer:
				answer = 0
				if ',,,,,,' in str(result):
					print('GPS is not ready')
					rec_null = False
					time.sleep(1)
				else:
						lat = str(result)[30:41]
						lon = str(result)[44:56]
						year = str(result)[59:61]
						month = str(result)[61:63]
						day = str(result)[63:65]
						hour = str(result)[66:68]
						minute = str(result)[68:70]
						second = str(result)[70:72]
						return lat,lon,year,month,day,hour,minute,second
			else:
				print('error %d'%answer)
				rec_buff = ''
				self.send_at('AT+CGPS=0','OK',1)
				return False
			time.sleep(1.5)


	def power_on(self,power_key):
		print('SIM7600X is starting:')
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(power_key,GPIO.OUT)
		time.sleep(0.1)
		GPIO.output(power_key,GPIO.HIGH)
		time.sleep(2)
		GPIO.output(power_key,GPIO.LOW)
		time.sleep(20)
		ser.flushInput()
		print('SIM7600X is ready')

	def power_down(self,power_key):
		print('SIM7600X is loging off:')
		GPIO.output(power_key,GPIO.HIGH)
		time.sleep(3)
		GPIO.output(power_key,GPIO.LOW)
		time.sleep(18)
		print('Good bye')