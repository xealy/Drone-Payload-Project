import os
from flask import Blueprint, send_from_directory, render_template, Response, request, redirect
from .models import DataModel, ImageModel, MeasurementChart
from . import db
from sqlalchemy import inspect, desc, asc
from datetime import datetime
# from .forms import TimeRangeForm

# Camera imports
import cv2
import depthai as dai
import time

# TAIP imports
from pathlib import Path
import sys
import numpy as np
import argparse
import json
import blobconverter


bp = Blueprint('main', __name__)


# capture_interval = 2  # Set this to your desired interval

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


@bp.route('/', methods=['GET', 'POST'])
def index():
    latest_data = db.session.query(DataModel).order_by(desc(DataModel.timestamp)).first()
    data = db.session.query(DataModel)

    NewChart = MeasurementChart()
    NewChart.data.label = "Air Quality"

    labels_array = []
    nh3_values_array = []
    ox_values_array = []
    red_values_array = []

    for record in data:
        labels_array.append(record.timestamp.strftime("%m/%d/%Y, %H:%M:%S"))
        red_values_array.append(record.reducing_gases)
        ox_values_array.append(record.oxidising_gases)
        nh3_values_array.append(record.ammonia_gases)

    NewChart.set_labels(labels_array)
    NewChart.set_data('Ammonia', nh3_values_array)
    NewChart.set_data('OX', ox_values_array)
    NewChart.set_data('RED', red_values_array)

    ChartJSON = NewChart.get()

    return render_template('air_sampling.html', latest_data=latest_data, data=data, chartJSON=ChartJSON)


@bp.route('/target_detection', methods=['GET', 'POST'])
def target_detection():
    data = db.session.query(DataModel)
    images = db.session.query(ImageModel)

    return render_template('target_detection.html', data=data, images=images)


@bp.route('/data_logs', methods=['GET', 'POST'])
def data_logs():
    data = db.session.query(DataModel)

    # data_selection = request.args.get('id') # to get url query params

    # EXAMPLE FORM
    # categoryForm = DropdownForm()
    # selection = categoryForm.category.data
    # if selection == "All Categories":
    #     models = db.session.query(MLModel).order_by(asc(MLModel.category))
    # else:
    #     models = db.session.query(MLModel).filter(MLModel.category == categoryForm.category.data).order_by(asc(MLModel.category))
    # return render_template('data_logs.html', data=data, form=categoryForm)

    # form = TimeRangeForm()
    # if form.validate_on_submit():
    #     from_time = form.from_time.data
    #     to_time = form.to_time.data
    #     # Process the form data as needed
    #     # ~~~

    return render_template('data_logs.html', data=data)


@bp.route('/system_logs', methods=['GET', 'POST'])
def system_logs():
    data = db.session.query(DataModel)

    # models = db.session.query(MLModel).order_by(asc(MLModel.category))
    # models = db.session.query(MLModel).filter(MLModel.category == categoryForm.category.data).order_by(asc(MLModel.category))

    return render_template('system_logs.html', data=data)


@bp.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


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
            frame_with_bbox = jpeg.tobytes()

        # time.sleep(capture_interval)

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_with_bbox + b'\r\n\r\n')


@bp.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')