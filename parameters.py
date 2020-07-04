import time
from fractions import Fraction
import cv2
import subprocess

def setparams(c):
    c.set(cv2.CAP_PROP_FPS, 90)
    c.set(cv2.CAP_PROP_EXPOSURE, 15)
    c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(28,0) #Focus
    command = "v4l2-ctl -d 0 -c compression_quality=1 -c white_balance_auto_preset=0"
    command += " -c iso_sensitivity_auto=0 -c iso_sensitivity=4"
    output = subprocess.call(command, shell=True)
    time.sleep(2)

def getparams(c):
    print("FPS set at : " + str(c.get(cv2.CAP_PROP_FPS)))
    print("Auto Exposure : " + str(c.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    print("Exposure time in ms : " + str(c.get(cv2.CAP_PROP_EXPOSURE)))
    print("Resolution Width : " + str(c.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Resolution Height : " + str(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
