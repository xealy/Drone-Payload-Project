import os
from flask import Blueprint, send_from_directory, render_template, Response, request, redirect, jsonify, url_for, make_response, current_app
from .models import DataModel, ImageModel, MeasurementChart
from . import db
from sqlalchemy import inspect, desc, asc, and_
from datetime import datetime, date
import requests

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
import base64


bp = Blueprint('main', __name__)


# START OF TAIP CONFIG
# Parse config
configPath = Path('/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/taip_assets/best.json')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# Parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# Extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# Parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# Get model path
nnPath = '/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/taip_assets/best_openvino_2022.1_6shave.blob'
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo('best_openvino_2022.1_6shave.blob', shaves = 6, zoo_type = "depthai", use_cache=True))

# Sync outputs
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
# *** Global variable for Image Stream
lastSavedTime = time.monotonic() # ALEX ADDED THIS
# END OF TAIP CONFIG


@bp.route('/', methods=['GET', 'POST'])
def index():
    def make_chart(data):
        # Create a new instance of MeasurementChart
        NewChart = MeasurementChart()
        NewChart.data.label = "Air Quality"

        # Initialize arrays for labels and data values
        labels_array = []
        nh3_values_array = []
        ox_values_array = []
        red_values_array = []

        # Populate arrays with data from the records
        for record in data:
            labels_array.append(record.timestamp.strftime("%m/%d/%Y, %H:%M:%S"))
            red_values_array.append(record.reducing_gases)
            ox_values_array.append(record.oxidising_gases)
            nh3_values_array.append(record.ammonia_gases)

        # Set labels and data for the chart
        NewChart.set_labels(labels_array)
        NewChart.set_data('Ammonia', nh3_values_array)
        NewChart.set_data('OX', ox_values_array)
        NewChart.set_data('RED', red_values_array)

        # Get the JSON representation of the chart
        ChartJSON = NewChart.get()

        return ChartJSON

    if request.method == 'POST':
        print("we got a post request :))")
        json_data = request.get_json()
        if json_data:
            print(f"Received data: {json_data}")

        # Extract data from the request
        timestamp = datetime.strptime(json_data['timestamp'], '%d/%m/%Y %H:%M:%S')
        reducing_gases = json_data['reducing_gases']
        oxidising_gases = json_data['oxidising_gases']
        ammonia_gases = json_data['nh3_gases']
        temperature = json_data['temperature']
        humidity = json_data['humidity']
        air_pressure = json_data['pressure']
        lux = json_data['light']

        # Create a new DataModel instance
        new_record = DataModel(
            timestamp=timestamp,
            reducing_gases=reducing_gases,
            oxidising_gases=oxidising_gases,
            ammonia_gases=ammonia_gases,
            temperature=temperature,
            humidity=humidity,
            air_pressure=air_pressure,
            lux=lux,
        )

        # Add the record to the session and commit
        try:
            db.session.add(new_record)
            db.session.commit()
            print("Record added successfully")
        except Exception as e:
            db.session.rollback()
            print("An error occurred")

    latest_data = db.session.query(DataModel).order_by(desc(DataModel.timestamp)).first()
    data = db.session.query(DataModel)

    ChartJSON = make_chart(data)

    return render_template('air_sampling.html', latest_data=latest_data, data=data, chartJSON=ChartJSON)


@bp.route('/target_detection', methods=['GET', 'POST'])
def target_detection():
    if request.method == 'POST':
        print("we got a post request :))")
        json_data = request.get_json()

        # if json_data:
        #     print(f"Received data: {json_data['image_bytestring_encoded']}")

        # Extract data from JSON object
        timestamp = datetime.strptime(json_data['timestamp'], '%d/%m/%Y %H:%M:%S')
        image_path = json_data['image_path']
        image_bytestring = json_data['image_bytestring_encoded']
        # image_bytestring = base64.b64decode(image_bytestring_encoded)

        # Create a new ImageModel instance
        new_record = ImageModel(
            timestamp=timestamp,
            image_path=image_path,
            image_bytestring=image_bytestring
        )

        # Add the record to the session and commit
        try:
            db.session.add(new_record)
            db.session.commit()
            print("Record added successfully")
        except Exception as e:
            db.session.rollback()
            print("An error occurred")
        
        # Redirect to the same route to trigger a GET request
        return redirect(url_for('main.target_detection'))

    data = db.session.query(DataModel)
    images = db.session.query(ImageModel).order_by(desc(ImageModel.timestamp))

    return render_template('target_detection.html', data=data, images=images)


def get_frame():
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    global lastSavedTime
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

            # ALEX ADDED THIS: save image every 4 seconds (for image stream)
            currentTime = time.monotonic()
            if currentTime - lastSavedTime >= 4:
                current_datetime = datetime.now()
                currentDatetimeFile = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

                current_datetime_string = current_datetime.strftime("%d/%m/%Y %H:%M:%S") # for db record
                new_image_to_serve = f'image_stream/{currentDatetimeFile}.jpg' # for db record
                frame_bytestring = jpeg.tobytes() # for db record
                frame_bytestring_encoded = base64.b64encode(frame_bytestring).decode('utf-8')

                new_image = f'/home/455Team/Documents/EGH455-UAV-Project/ui/dashboard/static/image_stream/{currentDatetimeFile}.jpg'
                # cv2.imwrite(new_image, frame)
                lastSavedTime = currentTime

                # SEND POST REQUEST to 'target_detection' endpoint
                data = {
                    "timestamp": current_datetime_string,
                    "image_path": new_image_to_serve,
                    "image_bytestring_encoded": frame_bytestring_encoded
                }
                response = requests.post("http://127.0.0.1:5000/target_detection", json=data)
                if response.status_code == 200:
                    print("Data posted successfully.")
                else:
                    print(f"Failed to post data. Status code: {response.status_code}")
                
            frame_with_bbox = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_with_bbox + b'\r\n\r\n')


@bp.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/data_logs', methods=['GET', 'POST'])
def data_logs():
    if request.method == 'POST':
        print("we got a post request :))")

        reference_date = date.today()
        
        # Get the time strings from the form
        from_time_str = request.form.get('from_time')
        to_time_str = request.form.get('to_time')
        
        # Convert time strings to time objects
        from_time = datetime.strptime(from_time_str, '%I:%M %p').time()
        to_time = datetime.strptime(to_time_str, '%I:%M %p').time()
        
        # Combine date and time into datetime objects
        from_datetime = datetime.combine(reference_date, from_time)
        to_datetime = datetime.combine(reference_date, to_time)

        data = db.session.query(DataModel).filter(and_(DataModel.timestamp >= from_datetime, DataModel.timestamp <= to_datetime)).all()
        images = db.session.query(ImageModel).filter(and_(ImageModel.timestamp >= from_datetime, ImageModel.timestamp <= to_datetime)).all()

        # convert results to dictionary with timestamp as the key
        data_dict = {item.timestamp: item for item in data}
        images_dict = {item.timestamp: item for item in images}

        # create list to store merged results
        merged_results = []

        # merge data based on timestamp
        for timestamp in sorted(set(data_dict.keys()).union(images_dict.keys())):
            merged_entry = {
                'timestamp': timestamp,
                'data': data_dict.get(timestamp),
                'image': images_dict.get(timestamp)
            }
            merged_results.append(merged_entry)

        return render_template('data_logs.html', data=merged_results)

    # Filter records based on the time frame
    data = db.session.query(DataModel).all()
    images = db.session.query(ImageModel).all()

    # convert results to dictionary with timestamp as the key
    data_dict = {item.timestamp: item for item in data}
    images_dict = {item.timestamp: item for item in images}

    # create list to store merged results
    merged_results = []

    # merge data based on timestamp
    for timestamp in sorted(set(data_dict.keys()).union(images_dict.keys())):
        merged_entry = {
            'timestamp': timestamp,
            'data': data_dict.get(timestamp),
            'image': images_dict.get(timestamp)
        }
        merged_results.append(merged_entry)

    return render_template('data_logs.html', data=merged_results)


@bp.route('/system_logs', methods=['GET', 'POST'])
def system_logs():
    data = db.session.query(DataModel).all()
    images = db.session.query(ImageModel).all()

    # convert results to dictionary with timestamp as the key
    data_dict = {item.timestamp: item for item in data}
    images_dict = {item.timestamp: item for item in images}

    # create list to store merged results
    merged_results = []

    # merge data based on timestamp
    for timestamp in sorted(set(data_dict.keys()).union(images_dict.keys())):
        merged_entry = {
            'timestamp': timestamp,
            'data': data_dict.get(timestamp),
            'image': images_dict.get(timestamp)
        }
        merged_results.append(merged_entry)

    return render_template('system_logs.html', data=merged_results)


@bp.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


# @bp.route('/static/<path:filename>')
# def serve_static(filename):
#     response = make_response(send_from_directory('static', filename))
#     response.headers['Cache-Control'] = 'public, max-age=31536000'  # Cache for one year
#     return response
