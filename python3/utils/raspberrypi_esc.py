import numpy as np
import os
import time
os.system ("sudo pigpiod") #Launching GPIO library
time.sleep(1) # If this delay is removed you will get an error
import pigpio

class raspi_esc:
    def __init__(self):
        self.ESC0=13  #Connect the ESC in this GPIO pin
        self.ESC1=12  #Connect the ESC in this GPIO pin
        self.pi = pigpio.pi()
        self.pi.set_servo_pulsewidth(self.ESC0, 0)
        self.pi.set_servo_pulsewidth(self.ESC1, 0)

        self.max_value = 2000 #ESC's max value
        self.min_value = 1000  #ESC's min value
        self.center_value = 1500

        self.arm()

    def calibrate(self):   #This is the auto calibration procedure of a normal ESC
        self.pi.set_servo_pulsewidth(self.ESC0, 0)
        self.pi.set_servo_pulsewidth(self.ESC1, 0)
        print("Disconnect the battery and press Enter")
        inp = input()
        if inp == '':
            self.pi.set_servo_pulsewidth(self.ESC0, self.max_value)
            self.pi.set_servo_pulsewidth(self.ESC1, self.max_value)
            print("Connect the battery NOW. You will here two beeps, then wait for a gradual falling tone then press Enter")
            inp = input()
            if inp == '':
                self.pi.set_servo_pulsewidth(self.ESC0, self.min_value)
                self.pi.set_servo_pulsewidth(self.ESC1, self.min_value)
                time.sleep(10)
                print('Calibration done')
                self.arm()

    def arm(self): #This is the arming procedure of an ESC
        self.control(0)
        time.sleep(2)
        print("ESC: armed")

    def control(self,u):
        u = self.transform(u)
        pwm = self.center_value + max((self.max_value - self.center_value)*u,0) + min((self.center_value - self.min_value)*u,0)
        self.pi.set_servo_pulsewidth(self.ESC0, pwm)
        self.pi.set_servo_pulsewidth(self.ESC1, pwm)

    def stop(self): #This will stop every action your Pi is performing for ESC
        self.pi.set_servo_pulsewidth(self.ESC0, 0)
        self.pi.set_servo_pulsewidth(self.ESC1, 0)
        self.pi.stop()
        print("ESC: motor stopped")

    def transform(self,u):
        s = np.sign(u)
        u = abs(u)
        # following values from calibrate_esc.py
        a,b,c,d,e,f,g = [-0.02151806, -0.28606846, 0.65830371, -0.03707646, 3.88293423, -0.02374704, 6.47285477]
        return s*abs(a + b*u**c + d*u**e + f*u**g)
