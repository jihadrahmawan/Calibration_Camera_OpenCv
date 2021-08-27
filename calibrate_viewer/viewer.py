import cv2
import numpy as np
from math import tan, pi
import glob
import yaml

if __name__ == '__main__':
    file_ = open("calibraton_data_ELPUSBFHD01M-BFV_v2.yaml")
    present = yaml.load(file_, Loader=yaml.FullLoader)
    mtxs=np.asarray(present["mtx"])
    dists=np.asarray(present["dist"])
    cap = cv2.VideoCapture(0)
    width = 640
    height= 480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    try: 
        while (1):
            ret, frame = cap.read()
            if ret:
                hh,  ww = frame.shape[:2]
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtxs,dists,(ww,hh),1,(ww,hh))
                x,y,w,h = roi
                dst = cv2.undistort(frame, mtxs, dists, None, newcameramtx)
                flat_image = dst[y:y+h, x:x+w]
                cv2.imshow("result", flat_image)
                cv2.imshow("original", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
    
    finally:
        cap.release()  
        cv2.destroyAllWindows()
