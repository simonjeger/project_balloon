#!/usr/bin/python

import RPi.GPIO as GPIO
import serial
import time
import datetime

class raspi_com():
	def __init__(self,phone_number):
		self.ser = serial.Serial("/dev/ttyUSB2",115200)
		self.ser.flushInput()

		self.phone_number = phone_number #********** change it to the phone number you want to text
		self.power_key = 6
		self.rec_buff = ''

		try:
			self.power_on()
			self.send_sms('Communication Initialized')
		except :
			if self.ser != None:
				ser.close()
			GPIO.cleanup()

	def send_at(self,command,back,timeout):
		self.rec_buff = ''
		self.ser.write((command+'\r\n').encode())
		time.sleep(timeout)
		if self.ser.inWaiting():
			time.sleep(0.01 )
			self.rec_buff = self.ser.read(self.ser.inWaiting())
		if back not in self.rec_buff.decode():
			print('COM: ' + command + ' ERROR')
			print('COM: ' + command + ' back:\t' + self.rec_buff.decode())
			return 0
		else:
			return 1

	def send_sms(self,text_message):
		self.send_at("AT+CMGF=1","OK",1)
		answer = self.send_at("AT+CMGS=\""+self.phone_number+"\"",">",2)
		if 1 == answer:
			self.ser.write(text_message.encode())
			self.ser.write(b'\x1A')
			answer = self.send_at('','OK',20)
			if 1 == answer:
				print("COM: sent SMS successfully")
			else:
				print('COM: sending error')
		else:
			print('COM: sending error%d'%answer)

	def receive_sms(self):
		self.rec_buff = ''
		self.send_at('AT+CMGF=1','OK',1)
		self.send_at('AT+CPMS=\"SM\",\"SM\",\"SM\"', 'OK', 1)
		self.send_at('AT+CPMS=\"SM\",\"SM\",\"SM\"', 'OK', 1)
		answer = self.send_at('AT+CMGR=1','+CMGR:',2)
		if 1 == answer:
			answer = 0
			if 'OK'.encode('utf-8') in self.rec_buff:
				answer = 1
				result = str(self.rec_buff).split("\\")[-7][1::]
				year = 2000 + int(str(self.rec_buff).split("\\")[-9][-21:-19])
				month = int(str(self.rec_buff).split("\\")[-9][-18:-16])
				day = int(str(self.rec_buff).split("\\")[-9][-15:-13])
				hour = int(str(self.rec_buff).split("\\")[-9][-12:-10])
				minute = int(str(self.rec_buff).split("\\")[-9][-9:-7])
				second = int(str(self.rec_buff).split("\\")[-9][-6:-4])

				timestamp = datetime.datetime(year,month,day,hour,minute,second)
				return result, timestamp
		else:
			print('COM: receiving error%d'%answer)
			return False
		return True

	def power_on(self):
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(self.power_key,GPIO.OUT)
		time.sleep(0.1)
		GPIO.output(self.power_key,GPIO.HIGH)
		time.sleep(2)
		GPIO.output(self.power_key,GPIO.LOW)
		time.sleep(20)
		self.ser.flushInput()
		print('COM: powered on')

	def power_off(self):
		GPIO.output(self.power_key,GPIO.HIGH)
		time.sleep(3)
		GPIO.output(self.power_key,GPIO.LOW)
		time.sleep(18)
		print('COM: powered off')
