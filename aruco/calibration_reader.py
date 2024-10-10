#!/usr/bin/env python3

import depthai as dai
import numpy as np
import sys
from pathlib import Path

# Camera A is the RGB Camera which is being used
# This code obtains the intrinsics defaults since resolution isn't set
# It also obtains the distortion coefficient values
with dai.Device() as device:
    calibData = device.readCalibration()

    M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)

    print("RGB Camera Default Intrinsics (Camera A):")
    print(np.array(M_rgb))
    print(f"Resolution: {width} x {height}")

    # Print the distortion coefficients if required
    D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
    print("RGB Distortion Coefficients:")
    [print(name + ": " + value) for (name, value) in
     zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
         [str(data) for data in D_rgb])]

    print(f'RGB FOV {calibData.getFov(dai.CameraBoardSocket.CAM_A)}')