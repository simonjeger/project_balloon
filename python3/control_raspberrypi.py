# GPS
import RPi.GPIO as GPIO

import serial
import time

ser = serial.Serial('/dev/ttyUSB2',115200)
ser.flushInput()

power_key = 6
rec_buff = ''
rec_buff2 = ''
time_count = 0

def send_at(command,back,timeout):
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
			return 0
		else:
			print(rec_buff.decode())
			return 1
	else:
		print('GPS is not ready')
		return 0

def get_gps_position():
	rec_null = True
	answer = 0
	print('Start GPS session...')
	rec_buff = ''
	send_at('AT+CGPS=1,1','OK',1)
	time.sleep(2)
	while rec_null:
		answer = send_at('AT+CGPSINFO','+CGPSINFO: ',1)
		if 1 == answer:
			answer = 0
			if ',,,,,,' in rec_buff:
				print('GPS is not ready')
				rec_null = False
				time.sleep(1)
		else:
			print('error %d'%answer)
			rec_buff = ''
			send_at('AT+CGPS=0','OK',1)
			return False
		time.sleep(1.5)


def power_on(power_key):
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

def power_down(power_key):
	print('SIM7600X is loging off:')
	GPIO.output(power_key,GPIO.HIGH)
	time.sleep(3)
	GPIO.output(power_key,GPIO.LOW)
	time.sleep(18)
	print('Good bye')

try:
	power_on(power_key)
	get_gps_position()
	power_down(power_key)
except:
	if ser != None:
		ser.close()
	power_down(power_key)
	GPIO.cleanup()
if ser != None:
		ser.close()
		GPIO.cleanup()

# Telecom Communication
#!/usr/bin/python

import RPi.GPIO as GPIO
import serial
import time

ser = serial.Serial('/dev/ttyUSB2',115200)
ser.flushInput()

power_key = 6
rec_buff = ''
APN = 'CMNET'
ServerIP = '118.190.93.84'
Port = '2317'
Message = 'Waveshare'

def power_on(power_key):
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

def power_down(power_key):
	print('SIM7600X is loging off:')
	GPIO.output(power_key,GPIO.HIGH)
	time.sleep(3)
	GPIO.output(power_key,GPIO.LOW)
	time.sleep(18)
	print('Good bye')

def send_at(command,back,timeout):
	rec_buff = ''
	ser.write((command+'\r\n').encode())
	time.sleep(timeout)
	if ser.inWaiting():
		time.sleep(0.1 )
		rec_buff = ser.read(ser.inWaiting())
	if rec_buff != '':
		if back not in rec_buff.decode():
			print(command + ' ERROR')
			print(command + ' back:\t' + rec_buff.decode())
			return 0
		else:
			print(rec_buff.decode())
			return 1
	else:
		print(command + ' no responce')

try:
	power_on(power_key)
	send_at('AT+CSQ','OK',1)
	send_at('AT+CREG?','+CREG: 0,1',1)
	send_at('AT+CPSI?','OK',1)
	send_at('AT+CGREG?','+CGREG: 0,1',0.5)
	send_at('AT+CGSOCKCONT=1,\"IP\",\"'+APN+'\"','OK',1)
	send_at('AT+CSOCKSETPN=1', 'OK', 1)
	send_at('AT+CIPMODE=0', 'OK', 1)
	send_at('AT+NETOPEN', '+NETOPEN: 0',5)
	send_at('AT+IPADDR', '+IPADDR:', 1)
	send_at('AT+CIPOPEN=0,\"TCP\",\"'+ServerIP+'\",'+Port,'+CIPOPEN: 0,0', 5)
	send_at('AT+CIPSEND=0,', '>', 2)#If not sure the message number,write the command like this: AT+CIPSEND=0, (end with 1A(hex))
	ser.write(Message.encode())
	if 1 == send_at(b'\x1a'.decode(),'OK',5):
		print('send message successfully!')
	send_at('AT+CIPCLOSE=0','+CIPCLOSE: 0,0',15)
	send_at('AT+NETCLOSE', '+NETCLOSE: 0', 1)
	power_down(power_key)
except:
	if ser != None:
		ser.close()
		GPIO.cleanup()

if ser != None:
		ser.close()
		GPIO.cleanup()

# ESC control
import os
import time
os.system ("sudo pigpiod") #Launching GPIO library
time.sleep(1) # If this delay is removed you will get an error
import pigpio

ESC=13  #Connect the ESC in this GPIO pin

pi = pigpio.pi()
pi.set_servo_pulsewidth(ESC, 0)

max_value = 2000 #ESC's max value
min_value = 700  #ESC's min value

def calibrate():   #This is the auto calibration procedure of a normal ESC
    pi.set_servo_pulsewidth(ESC, 0)
    print("Disconnect the battery and press Enter")
    inp = input()
    if inp == '':
        pi.set_servo_pulsewidth(ESC, max_value)
        print("Connect the battery NOW. You will here two beeps, then wait for a gradual falling tone then press Enter")
        inp = input()
        if inp == '':
            pi.set_servo_pulsewidth(ESC, min_value)
            pi.set_servo_pulsewidth(ESC, 0)
            time.sleep(2)
            print("Arming ESC now")
            pi.set_servo_pulsewidth(ESC, min_value)
            time.sleep(1)
            print("Armed")
            control() # You can change this to any other function you want

def arm(): #This is the arming procedure of an ESC
    if inp == '':
        pi.set_servo_pulsewidth(ESC, 0)
        time.sleep(1)
        pi.set_servo_pulsewidth(ESC, max_value)
        time.sleep(1)
        pi.set_servo_pulsewidth(ESC, min_value)
        time.sleep(1)
        control()

def control(u):
    pwm_low = 700
    pwm_high = 2000
    pwm = pwm_low + (pwm_high - pwm_low)*u

    pi.set_servo_pulsewidth(ESC, pwm)

def stop(): #This will stop every action your Pi is performing for ESC ofcourse.
    pi.set_servo_pulsewidth(ESC, 0)
    pi.stop()

#calibrate()
arm()
control(0.5)
stop()
