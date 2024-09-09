from flask import Blueprint, send_from_directory, render_template, request, redirect
from .models import DataModel, MeasurementChart
from . import db
from sqlalchemy import desc, asc
from .forms import TimeRangeForm

bp = Blueprint('main', __name__)


@bp.route('/', methods=['GET', 'POST'])
def index():
    latest_data = db.session.query(DataModel).order_by(desc(DataModel.timestamp)).first()
    data = db.session.query(DataModel)
    # data_selection = request.args.get('id')  # to get url query params
    # print(data[0].timestamp)

    NewChart = MeasurementChart()
    NewChart.data.label = "Air Quality"
    NewChart.set_labels(['2024-08-30 10:00:00', '2024-08-30 10:00:04', '2024-08-30 10:00:08', '2024-08-30 10:00:12', '2024-08-30 10:00:16', '2024-08-30 10:00:20', '2024-08-30 10:00:24'])
    NewChart.set_data('Ammonia', [0.12, 0.11, 0.12, 0.13, 0.12, 0.14, 0.15])
    NewChart.set_data('OX', [0.25, 0.24, 0.22, 0.21, 0.22, 0.23, 0.23])
    NewChart.set_data('RED', [3.2, 3.3, 3.3, 3.4, 3.3, 3.4, 3.5])
    ChartJSON = NewChart.get()

    return render_template('air_sampling.html', latest_data=latest_data, data=data, chartJSON=ChartJSON)


@bp.route('/target_detection', methods=['GET', 'POST'])
def target_detection():
    data = db.session.query(DataModel)
    # data_selection = request.args.get('id') # to get url query params

    return render_template('target_detection.html', data=data)


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
    # data_selection = request.args.get('id') # to get url query params

    # models = db.session.query(MLModel).order_by(asc(MLModel.category))
    # models = db.session.query(MLModel).filter(MLModel.category == categoryForm.category.data).order_by(asc(MLModel.category))

    return render_template('system_logs.html', data=data)


@bp.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

