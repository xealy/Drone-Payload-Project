# Camera imports
import cv2
import depthai as dai
import time
from datetime import datetime

# TAIP imports
from pathlib import Path
import sys
import numpy as np
import argparse
import json
import blobconverter

import requests
import base64


# NEW TAIP CONFIG
# parse config
configPath = Path('/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/best.json')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
# nnPath = args.model
nnPath = '/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/best_openvino_2022.1_6shave.blob'
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo('best_openvino_2022.1_6shave.blob', shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# *** Connect to device
device = dai.Device(pipeline)
# END OF: NEW TAIP CONFIG


def get_frame():
    # # Diagnostic print statements
    # print('Connected cameras:', device.getConnectedCameraFeatures())
    # print('Usb speed:', device.getUsbSpeed().name)
    # if device.getBootloaderVersion() is not None:
    #     print('Bootloader version:', device.getBootloaderVersion())
    # print('Device name:', device.getDeviceName())

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    lastSavedTime = startTime # ALEX ADDED THIS
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame, detections)
            _, jpeg = cv2.imencode('.jpg', frame)

            # ALEX ADDED THIS: save image every 2 seconds (for image stream)
            currentTime = time.monotonic()
            if currentTime - lastSavedTime >= 2:
                currentDatetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(f'/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/static/targets/{currentDatetime}.jpg', frame)
                lastSavedTime = currentTime

            # save detection as well

            # save all this to database

            frame_with_bbox = jpeg.tobytes()
            frame_bytestring = (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_with_bbox + b'\r\n\r\n')

            frame_base64_encoded = base64.b64encode(frame_bytestring).decode('utf-8')

        # POST REQUEST
        data = {
            "frame": frame_base64_encoded
        }

        try:
            response = requests.post("http://127.0.0.1:5000/video_feed", json=data)
            if response.status_code == 200:
                print("Data posted successfully.")
            else:
                print(f"Failed to post data. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error posting data: {e}")

# Call function
get_frame()

# FLASK ROUTE
# @bp.route('/video_feed', methods=['GET', 'POST'])
# def video_feed():
#     if request.method == 'POST':
#         print("we got a post request :))")

#         data = request.get_json()
#         base64_encoded_frame = data.get("frame")
        
#         if base64_encoded_frame:
#             # print(f"Received data: {base64_encoded_frame}")
#             frame_bytestring = base64.b64decode(base64_encoded_frame)

#         return Response(frame_bytestring, mimetype='multipart/x-mixed-replace; boundary=frame')
        
