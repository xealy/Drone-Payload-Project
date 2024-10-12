from . import db
from pychartjs import BaseChart, ChartType, Color, Options


class DataModel(db.Model):
    __tablename__ = 'data'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    reducing_gases = db.Column(db.String, nullable=True)
    oxidising_gases = db.Column(db.String, nullable=True)
    ammonia_gases = db.Column(db.String, nullable=True)
    temperature = db.Column(db.String, nullable=True)
    humidity = db.Column(db.String, nullable=True)
    air_pressure = db.Column(db.String, nullable=True)
    lux = db.Column(db.String, nullable=True)


class ImageModel(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    image_path = db.Column(db.String, nullable=True)
    coordinates = db.Column(db.String, nullable=True)
    valve_status = db.Column(db.String, nullable=True)
    gauge_reading = db.Column(db.String, nullable=True)
    image_bytestring = db.Column(db.Text, nullable=True)
    aruco_id = db.Column(db.String, nullable=True)


class MeasurementChart(BaseChart):
    type = ChartType.Line

    class labels:
        time = ['2024-08-30 10:00:00', '2024-08-30 10:00:04', '2024-08-30 10:00:08', '2024-08-30 10:00:12', '2024-08-30 10:00:16', '2024-08-30 10:00:20', '2024-08-30 10:00:24']

    class data:
        class Ammonia:
            label = "Ammonia (NH3)"
            type = ChartType.Line
            data = [80, 60, 100, 80, 90, 60, 80]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Purple

        class OX:
            label = "Oxidising Gases (OX)"
            type = ChartType.Line
            data = [60, 50, 80, 120, 140, 180, 170]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Green

        class RED:
            label = "Reducing Gases (RED)"
            type = ChartType.Line
            data = [90, 80, 60, 30, 50, 30, 20]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Red

    class options:
        title = Options.Title("Air Quality")
        scales = {
            "xAxes": [{
                "scaleLabel": {
                    "display": True,
                    "labelString": "Time"
                }
            }],
            "yAxes": [{
                # "id": "Ammonia",
                "position": "left",
                "ticks": {"beginAtZero": True},
                "scaleLabel": {
                    "display": True,
                    "labelString": "Gas Concentration (PPM)"
                }
            }]
        }

    def set_labels(self, new_labels):
        self.labels.time = new_labels

    def set_data(self, data_class, new_data):
        if hasattr(self.data, data_class):
            setattr(getattr(self.data, data_class), 'data', new_data)
        else:
            raise ValueError(f"No data class named {data_class}")
