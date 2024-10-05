import numpy as np
import cv2
import time
import depthai as dai
import numpy as np
import argparse
import time
from utils import ARUCO_DICT
import st7735

'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
# disp = st7735.ST7735(
#     port=0,
#     cs=1,
#     dc=9,
#     backlight=12,
#     rotation=270,
#     spi_speed_hz=10000000
# )

# # Initialize display
# disp.begin()
# WIDTH = disp.width
# HEIGHT = disp.height

# # Create pipeline
# pipeline = dai.Pipeline()

# # Define source and output
# camRgb = pipeline.create(dai.node.ColorCamera)
# xoutRgb = pipeline.create(dai.node.XLinkOut)

# xoutRgb.setStreamName("rgb")

# # Properties
# camRgb.setPreviewSize(WIDTH, HEIGHT)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# # Linking
# camRgb.preview.link(xoutRgb.input)

# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:
#     qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="/home/455Team/Documents/EGH455-UAV-Project/aruco/calibration_matrix.npy")
    ap.add_argument("-d", "--D_Coeff", required=True, help="/home/455Team/Documents/EGH455-UAV-Project/aruco/distortion_coefficients.npy")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="DICT_6X6_250")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()