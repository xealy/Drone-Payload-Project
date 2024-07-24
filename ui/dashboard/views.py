from flask import Blueprint, send_from_directory, render_template, request, redirect
from .forms import DropdownForm
from .models import MLModel, MeasurementChart
from . import db
from sqlalchemy import desc, asc

bp = Blueprint('main', __name__)


@bp.route('/', methods=['GET', 'POST'])
def index():
    # for reference
    # models = db.session.query(MLModel)
    # model_selection = request.args.get('id') # to get url query params

    categoryForm = DropdownForm()
    selection = categoryForm.category.data

    if selection == "All Categories":
        models = db.session.query(MLModel).order_by(asc(MLModel.category))
    else:
        models = db.session.query(MLModel).filter(MLModel.category == categoryForm.category.data).order_by(asc(MLModel.category))

    NewChart = MeasurementChart()
    NewChart.data.label = "Air Quality"
    ChartJSON = NewChart.get()

    return render_template('index.html', models=models, form=categoryForm, chartJSON=ChartJSON)


@bp.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

