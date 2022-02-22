import RPi.GPIO as GPIO
import serial
import time
import datetime
import pytz
from filelock import FileLock

import logging
logger=logging.getLogger()

class raspi_com():
	def __init__(self,phone_number,path):
		self.ser = serial.Serial("/dev/ttyUSB2",115200)
		self.ser.flushInput()

		self.phone_number = phone_number
		self.simcard = '0791747954' #sim card in balloon
		self.path = path
		self.power_key = 6
		self.rec_buff = ''

		try:
			self.power_on()
			self.delete_sms()
			self.send_sms('Communication initialized')
		except :
			if self.ser != None:
				self.ser.close()
			GPIO.cleanup()

	def send_at(self,command,back,timeout):
		self.rec_buff = ''
		self.ser.flush()
		self.ser.write((command+'\r\n').encode())
		time.sleep(timeout)
		if self.ser.inWaiting():
			time.sleep(0.01)
			self.rec_buff = self.ser.read(self.ser.inWaiting())
		if type(self.rec_buff) != str:
			if back not in self.rec_buff.decode(errors='ignore'):
				logger.error('COM: ' + command + ' ERROR')
				logger.error('COM: ' + command + ' back:\t' + self.rec_buff.decode())
				return 0
			else:
				return 1
		else:
			logger.error('COM: ' + command + ' string back:\t' + self.rec_buff)
			return 0

	def update(self, message):
		with FileLock(self.path + 'waveshare.lock'):
			self.send_sms_nfl(message)
			return self.receive_last_sms()

	def send_sms(self,text_message,phone_number=None):
		with FileLock(self.path + 'waveshare.lock'):
			text_message = text_message[0:160] #avoid sending too large messages which lead to an error
			if phone_number is None:
				phone_number = self.phone_number
			self.send_at("AT+CMGF=1","OK",1)
			answer = self.send_at("AT+CMGS=\""+phone_number+"\"",">",1.5)
			if 1 == answer:
				self.ser.write(text_message.encode())
				self.ser.write(b'\x1A')
				answer = self.send_at('','OK',1.5)
				if 1 == answer:
					logger.info("COM: Sent SMS successfully")
				else:
					logger.error('COM: Sending error')
			else:
				logger.error('COM: Sending error%d'%answer)

	def send_sms_nfl(self,text_message,phone_number=None):
		text_message = text_message[0:160] #avoid sending too large messages which lead to an error
		if phone_number is None:
			phone_number = self.phone_number
		self.send_at("AT+CMGF=1","OK",1)
		answer = self.send_at("AT+CMGS=\""+phone_number+"\"",">",1.5)
		if 1 == answer:
			self.ser.write(text_message.encode())
			self.ser.write(b'\x1A')
			answer = self.send_at('','OK',1.5)
			if 1 == answer:
				logger.info("COM: Sent SMS successfully")
			else:
				logger.error('COM: Sending error')
		else:
			logger.error('COM: Sending error%d'%answer)

	def receive_last_sms(self):
		self.rec_buff = ''
		#self.send_at("AT+CMGF=1","OK",1)
		#self.send_at('AT+CPMS=\"SM\",\"SM\",\"SM\"', 'OK', 1)
		answer = self.send_at('AT+CMGL="ALL"', '+CMGL:', 1.5)
		if 1 == answer:
			answer = 0
			if 'OK'.encode('utf-8') in self.rec_buff:
				answer = 1
				rec_string = str(self.rec_buff).split("\\")
				last_ID = int(rec_string[-9].split(',')[0][8::])

				result = rec_string[-7][1::]
				year = 2000 + int(rec_string[-9][-21:-19])
				month = int(rec_string[-9][-18:-16])
				day = int(rec_string[-9][-15:-13])
				hour = int(rec_string[-9][-12:-10])
				minute = int(rec_string[-9][-9:-7])
				second = int(rec_string[-9][-6:-4])
				timestamp = datetime.datetime(year,month,day,hour,minute,second)
				timezone = pytz.timezone("Europe/Zurich")
				timestamp = timezone.localize(timestamp)

				return result, timestamp
		else:
			logger.error('COM: Error in receive_last_sms')
			return None

	def receive_sms(self,ID):
		with FileLock(self.path + 'waveshare.lock'):
			self.rec_buff = ''
			self.send_at('AT+CMGF=1','OK',1)
			self.send_at('AT+CPMS=\"SM\",\"SM\",\"SM\"', 'OK', 1)
			answer = self.send_at('AT+CMGR=' + str(ID),'+CMGR:',1.5)
			if 1 == answer:
				answer = 0
				if 'OK'.encode('utf-8') in self.rec_buff:
					answer = 1
					rec_string = str(self.rec_buff).split("\\")

					result = rec_string[-7][1::]
					year = 2000 + int(rec_string[-9][-21:-19])
					month = int(rec_string[-9][-18:-16])
					day = int(rec_string[-9][-15:-13])
					hour = int(rec_string[-9][-12:-10])
					minute = int(rec_string[-9][-9:-7])
					second = int(rec_string[-9][-6:-4])

					timestamp = datetime.datetime(year,month,day,hour,minute,second)
					timezone = pytz.timezone("Europe/Zurich")
					timestamp = timezone.localize(timestamp)
					return result, timestamp
			else:
				logger.error('COM: Receiving error%d'%answer)
				return False
			return True

	def list_sms(self):
		with FileLock(self.path + 'waveshare.lock'):
			self.rec_buff = ''
			answer = self.send_at('AT+CMGL="ALL"', '+CMGL:', 1.5)
			if 1 == answer:
				answer = 0
				if 'OK'.encode('utf-8') in self.rec_buff:
					answer = 1
					rec_string = str(self.rec_buff).split("\\")
					last_ID = int(rec_string[-9].split(',')[0][8::])

					return last_ID
			else:
				logger.error('COM: Error in list_sms')
				return None

	def delete_sms(self):
		with FileLock(self.path + 'waveshare.lock'):
			self.rec_buff = ''
			answer = self.send_at('AT+CMGD=1,4', 'OK', 1.5)
			self.send_sms_nfl('inbox emptied', phone_number=self.simcard) #I can't read out an empty list
			logger.info('COM: inbox emptied')

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
			logger.info('COM: powered on')

	def power_off(self):
		with FileLock(self.path + 'waveshare.lock'):
			GPIO.output(self.power_key,GPIO.HIGH)
			time.sleep(3)
			GPIO.output(self.power_key,GPIO.LOW)
			time.sleep(18)
			logger.info('COM: powered off')
