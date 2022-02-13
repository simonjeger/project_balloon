from threading import Timer
import sys
import time
from utils.raspberrypi_esc import raspi_esc

esc = raspi_esc()
u = 0

while True:
    u = input('Enter input')
    try:
        u = float(u)
        esc.control(u)
        print('u = ' + str(u))
    except KeyboardInterrupt:
        break
    except:
        print("Couldn't convert to float")
esc.stop()
