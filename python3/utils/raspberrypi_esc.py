import os
import time
os.system ("sudo pigpiod") #Launching GPIO library
time.sleep(1) # If this delay is removed you will get an error
import pigpio

class raspi_esc:
    def __init__(self):
        self.ESC=13  #Connect the ESC in this GPIO pin
        pi = pigpio.pi()
        pi.set_servo_pulsewidth(self.ESC, 0)

        self.max_value = 2000 #ESC's max value
        self.min_value = 700  #ESC's min value

    def calibrate(self):   #This is the auto calibration procedure of a normal ESC
        pi.set_servo_pulsewidth(self.ESC, 0)
        print("Disconnect the battery and press Enter")
        inp = input()
        if inp == '':
            pi.set_servo_pulsewidth(self.ESC, self.max_value)
            print("Connect the battery NOW. You will here two beeps, then wait for a gradual falling tone then press Enter")
            inp = input()
            if inp == '':
                pi.set_servo_pulsewidth(self.ESC, self.min_value)
                pi.set_servo_pulsewidth(self.ESC, 0)
                time.sleep(2)
                print("Arming self.ESC now")
                pi.set_servo_pulsewidth(self.ESC, self.min_value)
                time.sleep(1)
                print("Armed")
                control() # You can change this to any other function you want

    def arm(self): #This is the arming procedure of an ESC
        pi.set_servo_pulsewidth(self.ESC, 0)
        time.sleep(1)
        pi.set_servo_pulsewidth(self.ESC, self.max_value)
        time.sleep(1)
        pi.set_servo_pulsewidth(self.ESC, self.min_value)
        time.sleep(1)
        control()

    def control(self,u):
        pwm_low = 700
        pwm_high = 2000
        pwm = pwm_low + (pwm_high - pwm_low)*u
        pi.set_servo_pulsewidth(self.ESC, pwm)

    def stop(self): #This will stop every action your Pi is performing for ESC
        pi.set_servo_pulsewidth(self.ESC, 0)
        pi.stop()
