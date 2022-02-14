from threading import Timer
import sys
import time
from utils.raspberrypi_esc import raspi_esc

esc = raspi_esc()
u = 0

while True:
    try:
        u = input('Enter input \n')
        u = float(u)
        esc.control(u)
        print('u = ' + str(u))
    except KeyboardInterrupt:
        print('Manual kill')
        break
    except:
        print("Couldn't convert to float")
esc.stop()
