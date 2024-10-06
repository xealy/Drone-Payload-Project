### Sample Code Only! 

import cv2
import numpy as np

# Load camera calibration parameters
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Replace with your camera matrix values
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Replace with your distortion coefficients

# Load the predefined dictionary of ArUco markers
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index to your camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate the pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length=0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        # Loop through each detected marker
        for i in range(len(ids)):
            # Draw the axis for each marker
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

            # Translation vector tvecs[i] gives (x, y, z) coordinates of the marker in the camera's frame of reference
            x, y, z = tvecs[i][0]

            print(f"Marker ID: {ids[i][0]} - Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
