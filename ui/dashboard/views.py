import os
from flask import Blueprint, send_from_directory, render_template, Response, request, redirect, jsonify, url_for, make_response, current_app
from .models import DataModel, ImageModel, MeasurementChart
from . import db
from sqlalchemy import inspect, desc, asc, and_
from datetime import datetime, date, timedelta
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
import math
import cv2.aruco as aruco

# LCD imports
from PIL import Image
from PIL import ImageDraw, ImageFont
import st7735
import colorsys
from fonts.ttf import RobotoMedium as UserFont
import socket

# Soil Sampling imports
import servo
from servo import ServoClass


bp = Blueprint('main', __name__)


# GlOBALS FOR LCD
lcd_mode_num = 0
lcd_variables = ["IP", "AQ", "TAIP"]
lcd_mode = lcd_variables[lcd_mode_num]

# Start LCD display
disp = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=1000000 # 1 MHz
)

# Initialize display
disp.begin()
WIDTH = disp.width
HEIGHT = disp.height

# Set up canvas and font (for AQ display)
img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
draw = ImageDraw.Draw(img)
path = os.path.dirname(os.path.realpath(__file__))
font_size = 20
font = ImageFont.truetype(UserFont, font_size)
message = ""
# The position of the top bar
top_pos = 25

# Create a values dict to store the data (for AQ display)
variables = ["temperature"]
values = {}
for v in variables:
    values[v] = [1] * WIDTH

def display_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connect to a host to retrieve the ipaddr
            s.connect(('8.8.8.8', 80))
            ipaddr = s.getsockname()[0]
        except Exception:
            # Catch undesired ipaddr results
            ipaddr = '127.0.0.1'
        finally:
            s.close()
        return ipaddr

    # New canvas to draw on.
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Text settings.
    font_size = 25
    font = ImageFont.truetype(UserFont, font_size)
    text_colour = (255, 255, 255)
    back_colour = (0, 170, 170)

    message = get_ip()

    x1, y1, x2, y2 = font.getbbox(message)
    size_x = x2 - x1
    size_y = y2 - y1

    # Calculate text position
    x = (WIDTH - size_x) / 2
    y = (HEIGHT / 2) - (size_y / 2)

    # Draw background rectangle and write text.
    draw.rectangle((0, 0, 160, 80), back_colour)
    draw.text((x, y), message, font=font, fill=text_colour)
    disp.display(img)

def display_lcd(frame):
    # Convert frame to PIL
    im_pil = Image.fromarray(frame)
    # Resize the image
    im_pil = im_pil.resize((WIDTH, HEIGHT))
    # Display image on LCD
    disp.display(im_pil)

def display_text(variable, data, unit):
    # Maintain length of list
    values[variable] = values[variable][1:] + [data]
    # Scale the values for the variable between 0 and 1
    vmin = min(values[variable])
    vmax = max(values[variable])
    colours = [(v - vmin + 1) / (vmax - vmin + 1) for v in values[variable]]
    # Format the variable name and value
    message = f"{variable[:4]}: {data:.1f} {unit}"
    draw.rectangle((0, 0, WIDTH, HEIGHT), (255, 255, 255))
    for i in range(len(colours)):
        # Convert the values to colours from red to blue
        colour = (1.0 - colours[i]) * 0.6
        r, g, b = [int(x * 255.0) for x in colorsys.hsv_to_rgb(colour, 1.0, 1.0)]
        # Draw a 1-pixel wide rectangle of colour
        draw.rectangle((i, top_pos, i + 1, HEIGHT), (r, g, b))
        # Draw a line graph in black
        line_y = HEIGHT - (top_pos + (colours[i] * (HEIGHT - top_pos))) + top_pos
        draw.rectangle((i, line_y, i + 1, line_y + 1), (0, 0, 0))
    # Write the text at the top in black
    draw.text((0, 0), message, font=font, fill=(0, 0, 0))
    disp.display(img)

# DISPLAY IP FIRST
display_ip()

# START OF ARUCO DEFINITIONS
# Camera specs obtained from calibration_reader.py
camera_matrix = np.array([[3.02075488e+03, 0.00000000e+00, 1.87725024e+03],
 [0.00000000e+00, 3.02075488e+03, 1.10085803e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([12.180327415466309, 7.460699081420898,
-8.580022404203191e-05, -0.0012392610078677535,
56.39138412475586])

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
# END OF ARUCO DEFINITIONS


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
    global lcd_mode # LCD GLOBAL
    global lcd_mode_num
    global lcd_variables

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

        # if json_data:
        #     print(f"Received data: {json_data}")

        # Extract data from the request
        timestamp = datetime.strptime(json_data['timestamp'], '%d/%m/%Y %H:%M:%S')
        reducing_gases = json_data['reducing_gases']
        oxidising_gases = json_data['oxidising_gases']
        ammonia_gases = json_data['nh3_gases']
        temperature = json_data['temperature']
        humidity = json_data['humidity']
        air_pressure = json_data['pressure']
        lux = json_data['light']
        change_lcd = json_data['change_lcd'] # for LCD toggle

        if change_lcd == 'True':
            
            # toggle LCD mode
            lcd_mode_num += 1
            lcd_mode_num %= len(lcd_variables)

            if lcd_mode_num == 0: # toggle to IP display
                lcd_mode = lcd_variables[lcd_mode_num]
                display_ip()
            if lcd_mode_num == 1: # toggle to AQ display
                lcd_mode = lcd_variables[lcd_mode_num]
            if lcd_mode_num == 2: # toggle to TAIP display
                lcd_mode = lcd_variables[lcd_mode_num]

        if lcd_mode == lcd_variables[1]:
            display_text(variables[0], temperature , "°C")

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
    
    # Get records (last 3 minutes from most recent reading)
    if latest_data is not None:
        latest_timestamp = latest_data.timestamp
        three_minutes_ago = latest_timestamp - timedelta(minutes=3)
        data_last_three_minutes = db.session.query(DataModel).filter(DataModel.timestamp >= three_minutes_ago).all()
        ChartJSON = make_chart(data_last_three_minutes)
    else: 
        data = db.session.query(DataModel)
        ChartJSON = make_chart(data)

    return render_template('air_sampling.html', latest_data=latest_data, chartJSON=ChartJSON)


@bp.route('/target_detection', methods=['GET', 'POST'])
def target_detection():
    global lcd_mode # LCD GLOBAL

    if request.method == 'POST':
        print("we got a post request :))")
        json_data = request.get_json()

        # Extract data from JSON object
        timestamp = datetime.strptime(json_data['timestamp'], '%d/%m/%Y %H:%M:%S')
        image_path = json_data['image_path']
        coordinates = json_data['coordinates']
        valve_status = json_data['valve_status']
        gauge_reading = json_data['gauge_reading']
        image_bytestring = json_data['image_bytestring_encoded']
        aruco_id = json_data['aruco_id']

        # Create a new ImageModel instance
        new_record = ImageModel(
            timestamp=timestamp,
            image_path=image_path,
            coordinates=coordinates,
            valve_status=valve_status,
            gauge_reading=gauge_reading,
            image_bytestring=image_bytestring,
            aruco_id=aruco_id
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
        # return redirect(url_for('main.target_detection'))

    # images = db.session.query(ImageModel).order_by(desc(ImageModel.timestamp))
    latest_images = db.session.query(ImageModel).order_by(desc(ImageModel.timestamp)).limit(100).all() # get latest 100 records ordered by timestamp in descending order

    # Display latest detections from each category
    latest_valve_status = db.session.query(ImageModel).filter(ImageModel.valve_status.isnot(None)).order_by(desc(ImageModel.timestamp)).first()
    latest_coordinates = db.session.query(ImageModel).filter(ImageModel.coordinates.isnot(None)).order_by(desc(ImageModel.timestamp)).first()
    latest_gauge_reading = db.session.query(ImageModel).filter(ImageModel.gauge_reading.isnot(None)).order_by(desc(ImageModel.timestamp)).first()

    return render_template('target_detection.html', images=latest_images , latest_valve_status=latest_valve_status, 
                           latest_coordinates=latest_coordinates, latest_gauge_reading=latest_gauge_reading)


def calculate_angle(base_point, tip_point):
    # Calculate the angle of the needle using the base and tip coordinates
    delta_y = tip_point[1] - base_point[1]
    delta_x = tip_point[0] - base_point[0]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    angle = round(angle)

    # Changes negative theta to appropriate value
    if angle < 0:
        angle *= -1
        angle = (180 - angle) + 180

    # Sets new starting point
    angle = angle - 90

    # Changes negative theta to appropriate value
    if angle < 0:
        angle *= -1
        angle = angle + 270

    if angle < 90:
        print("The air pressure gauge is less than 2 Bars! Trigger the motor!")
        servo_instance = ServoClass()
        servo_instance.start_servo() # run drill

    return angle


def map_angle_to_pressure(angle):
    # Linearly map the angle to the pressure range
    pressure = int((0.51 * angle) - 18.83)
    return pressure


def pose_estimation(frame, corners, ids, matrix_coefficients, distortion_coefficients):
    # Initialize a list to store the marker ID and pose information
    marker_positions = []
    
    for i in range(len(ids)):
        # Estimate pose of each marker
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
        
        # Draw Axis on the frame
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
        
        # Add the marker ID and its translation vector (x, y, z coordinates) to the list
        marker_info = {
            'id': ids[i],
            'x': tvec[0][0][0],
            'y': tvec[0][0][1],
            'z': tvec[0][0][2]
        }
        marker_positions.append(marker_info)
        
        # Optionally draw the position on the frame (for visualization)
        cv2.putText(frame, f"ID: {marker_info['id']} X: {marker_info['x']:.2f} Y: {marker_info['y']:.2f} Z: {marker_info['z']:.2f}", 
                    (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Return the annotated frame and the list of marker positions
    return frame, marker_positions


def detect_aruco(frame):
    (corners, ids, rejectedImgPoints) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if len(corners) > 0:
        ids = ids.flatten()
        corners_pos = corners
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
   
            # Draw a box around the ArUCO Marker
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
 
            # Display the Marker ID
            cv2.putText(frame, str(markerID),(topLeft[0], topLeft[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)

        frame, marker_positions = pose_estimation(frame, corners_pos, ids, camera_matrix, distortion_coefficients)

        return marker_positions


def get_frame():
    global lcd_mode # LCD GLOBAL

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
        tip = None
        base = None
        pressure = None
        valve_status = None
        aruco_marker = None
        marker_positions = None
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = labels[detection.label]

            # Gauge Reading
            if label == "Tip":
                tip = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            elif label == "Base":
                base = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            elif label == "BallValve_ON":
                valve_status = 'Open'
            elif label == "BallValve_OFF":
                valve_status = 'Closed'
            elif label == "ArUCO":
                aruco_marker = True

            # If both tip and base are detected, calculate the angle and pressure
            if tip is not None and base is not None:
                # Draw lines between tip and base
                cv2.line(frame, base, tip, (0, 255, 0), 2)

                # Calculate the angle of the needle
                angle = calculate_angle(base, tip)

                # Map the angle to a pressure reading
                pressure = map_angle_to_pressure(angle)

                # Display the pressure reading on the frame
                cv2.putText(frame, f"Pressure: {pressure} PSI", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        
        if aruco_marker is not None:
            marker_positions = detect_aruco(frame)

        return [pressure, valve_status, marker_positions]

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
            taip_detection_values = displayFrame("rgb", frame, detections)
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

                if lcd_mode == lcd_variables[2]:
                    display_lcd(frame)
                
                # SEND POST REQUEST to 'target_detection' endpoint
                if taip_detection_values[2] is not None:
                    x_coord = round(taip_detection_values[2][0]['x'], 3)
                    y_coord = round(taip_detection_values[2][0]['y'], 3)
                    z_coord = round(taip_detection_values[2][0]['z'], 3)

                    data = {
                        "timestamp": current_datetime_string,
                        "image_path": new_image_to_serve,
                        "coordinates": f"({x_coord}, {y_coord}, {z_coord})",
                        "valve_status": taip_detection_values[1],
                        "gauge_reading": taip_detection_values[0],
                        "image_bytestring_encoded": frame_bytestring_encoded, 
                        "aruco_id": str(taip_detection_values[2][0]['id'])
                    }
                else:
                    data = {
                        "timestamp": current_datetime_string,
                        "image_path": new_image_to_serve,
                        "coordinates": None,
                        "valve_status": taip_detection_values[1],
                        "gauge_reading": taip_detection_values[0],
                        "image_bytestring_encoded": frame_bytestring_encoded, 
                        "aruco_id": None
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
    validation_message = None

    # if no datetime range filter applied
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

    # sort merged results by timestamp in descending order
    merged_results = sorted(merged_results, key=lambda x: x['timestamp'], reverse=True)

    if request.method == 'POST':
        print("we got a post request :))")

        # get time strings from the form
        from_time_str = request.form.get('from_time')
        to_time_str = request.form.get('to_time')

        if not from_time_str or not to_time_str:
            validation_message = "Validation Error: Either 'From' field or 'To' field is empty"
            return render_template('data_logs.html', data=merged_results, validation_message=validation_message)
        
        # convert time strings to time objects
        from_time = datetime.strptime(from_time_str, '%I:%M %p').time()
        to_time = datetime.strptime(to_time_str, '%I:%M %p').time()

        if from_time > to_time:
            validation_message = "Validation Error: Ensure that 'From' field is earlier than 'To' field"
            return render_template('data_logs.html', data=merged_results, validation_message=validation_message)
        
        # get today's date
        reference_date = date.today()

        # combine date and time into datetime objects
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

        # sort merged results by timestamp in descending order
        merged_results = sorted(merged_results, key=lambda x: x['timestamp'], reverse=True)

        return render_template('data_logs.html', data=merged_results, validation_message=validation_message)

    return render_template('data_logs.html', data=merged_results, validation_message=validation_message)


@bp.route('/system_logs', methods=['GET', 'POST'])
def system_logs():
    data = None
    images = None

    # get records (last 10 minutes from most recent reading)
    latest_data = db.session.query(DataModel).order_by(desc(DataModel.timestamp)).first()
    if latest_data is not None:
        latest_timestamp = latest_data.timestamp
        ten_minutes_ago = latest_timestamp - timedelta(minutes=10)
        data = db.session.query(DataModel).filter(DataModel.timestamp >= ten_minutes_ago).all()
        images = db.session.query(ImageModel).filter(ImageModel.timestamp >= ten_minutes_ago).all()
    else: 
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
    
    # sort merged results by timestamp in descending order
    merged_results = sorted(merged_results, key=lambda x: x['timestamp'], reverse=True)

    return render_template('system_logs.html', data=merged_results)


@bp.route('/lcd_ip', methods=['GET'])
def lcd_ip():
    global lcd_mode
    global lcd_mode_num
    global lcd_variables

    lcd_mode_num = 0
    lcd_mode = lcd_variables[lcd_mode_num]
    if lcd_mode == lcd_variables[0]:
        display_ip()
    return '', 204  # No content


@bp.route('/lcd_temp', methods=['GET'])
def lcd_temp():
    global lcd_mode
    global lcd_mode_num
    global lcd_variables

    lcd_mode_num = 1
    lcd_mode = lcd_variables[lcd_mode_num]
    return '', 204  # No content


@bp.route('/lcd_feed', methods=['GET'])
def lcd_feed():
    global lcd_mode
    global lcd_mode_num
    global lcd_variables

    lcd_mode_num = 2
    lcd_mode = lcd_variables[lcd_mode_num]
    return '', 204  # No content


@bp.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
