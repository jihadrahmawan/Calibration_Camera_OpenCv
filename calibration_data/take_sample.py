# Import OpenCV and numpy
import cv2
import numpy as np

try: 
    index=0
    cap = cv2.VideoCapture(0)
    while (1):
        ret, frame = cap.read()
        if ret :
            cv2.imshow("image", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                index=index+1
                print ("SAVE!")
                str_index="%d_"%index
                filename = str_index+"file_.png"
                cv2.imwrite(filename, frame)
            if key == ord('q'):
                break
finally:
    cap.release()
    
