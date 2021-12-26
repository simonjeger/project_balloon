import time
import board
import adafruit_bmp3xx


class raspi_alt:
	def __init__(self):
		i2c = board.I2C()
		self.bmp = adafruit_bmp3xx.BMP3XX_I2C(i2c)

	def get_altitude(self):
		return self.bmp.altitude
