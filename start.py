from parameters import setparams,getparams
from detect import hit_detection
import signal
import cv2
import numpy as np
import time
import os




def signal_handler(signal, frame):
    global uninterrupted
    uninterrupted = False
    
saved = 1
uninterrupted = True
camera = cv2.VideoCapture(-1)
setparams(camera)
getparams(camera)
time.sleep(5)
signal.signal(signal.SIGINT, signal_handler)
os.system("reset")


frames = 0
start = time.time()
s = 0

while uninterrupted:
    ret, frame = camera.read()
    #img = np.array(frame)
    #print(img.shape)
    frames += 1
    #grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #print(img.argmax())
    brightest_pixel = np.max(frame)
    print('\r'+'Brightest Pixel :',brightest_pixel,'Saved :',saved-1,"    ",s,end='')
    #time.sleep(0.000001)
    if brightest_pixel > 60:
        s+=1
        non_zero = np.count_nonzero(frame)
        saved = hit_detection(frame,brightest_pixel,non_zero,saved)

camera.release()
print('\nTotal Number of frames :',frames)
print("Total time in seconds :", time.time()-start)
print("FPS :", frames/(time.time()-start))

        
