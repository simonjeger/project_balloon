from gpiozero import Servo
import time

servo = Servo(32)

servo.min()
time.sleep(3)
servo.mid()
time.sleep(3)
servo.max()
time.sleep(3)
