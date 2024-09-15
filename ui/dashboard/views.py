import os
from flask import Blueprint, send_from_directory, render_template, Response, request, redirect
from .models import DataModel, ImageModel, MeasurementChart
from . import db
from sqlalchemy import inspect, desc, asc
from datetime import datetime
# from .forms import TimeRangeForm

# for camera
import cv2
import depthai as dai
import time

bp = Blueprint('main', __name__)


# START OF: FOR CAMERA
# Create pipeline
pipeline = dai.Pipeline()

# Properties
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define source and output
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Connect to device
device = dai.Device(pipeline)
# END OF: FOR CAMERA


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
    print('Connected cameras:', device.getConnectedCameraFeatures())
    print('Usb speed:', device.getUsbSpeed().name)
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    print('Device name:', device.getDeviceName())

    while True:
        # Output queue will be used to get the RGB frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
        inRgb = qRgb.get()  # blocking call, will wait until new data has arrived
        image = inRgb.getCvFrame()

        _, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        # time.sleep(capture_interval)

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            


@bp.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')