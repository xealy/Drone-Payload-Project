import os
from flask import Blueprint, send_from_directory, render_template, request, redirect
from .models import DataModel, ImageModel, MeasurementChart
# from .models import MeasurementChart
from . import db
from sqlalchemy import inspect, desc, asc
# from .forms import TimeRangeForm
from datetime import datetime

bp = Blueprint('main', __name__)


@bp.route('/', methods=['GET', 'POST'])
def index():
    inspector = inspect(db.engine)
    table_names = inspector.get_table_names()
    print(table_names)

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

