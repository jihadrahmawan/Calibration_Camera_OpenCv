# Calibration_Camera_OpenCv
calibration camera example, has tested on ELP Camera ELP-USBFHD01M-BFV
using OpenCv python 4.2.1

how to use
1. run program Calibration_Camera_Opencv/calibration_data/take_sample.py
2. collect the data on your camera with chessboard, in this case use 9x6 
3. run Calibration_Camera_Opencv/calibration_data/calibration_procces.py
4. you will get file .yaml, that file is calibration matrix
5. move your calibration file.yaml to calibrate_viwer folder
6. run Calibration_Camera_Opencv/calibrate_viwer/viewer.py
