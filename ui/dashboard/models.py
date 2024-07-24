from . import db
from pychartjs import BaseChart, ChartType, Color, Options


class MLModel(db.Model):
    __tablename__ = 'models'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(255), nullable=False)
    img_src = db.Column(db.Text)


class MeasurementChart(BaseChart):
    type = ChartType.Bar

    class labels:
        time = list(range(1, 8))

    class data:
        class C02:
            label = "C02"
            type = ChartType.Line
            data = [80, 60, 100, 80, 90, 60, 80]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Purple
            # yAxisID = 'C02'

        class Temperature:
            label = "Temperature"
            type = ChartType.Line
            data = [60, 50, 80, 120, 140, 180, 170]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Green
            # yAxisID = 'Temperature'

        class Humidity:
            label = "Humidity"
            type = ChartType.Line
            data = [90, 80, 60, 30, 50, 30, 20]
            backgroundColor = Color.RGBA(0, 0, 0, 0)
            borderColor = Color.Red
            # yAxisID = 'Humidity'

    class options:
        title = Options.Title("Apples I've eaten compared to total daily energy")
        scales = {
            "yAxes": [
                {"id": "co2",
                 "position": "right",
                 "ticks": {"beginAtZero": True}
                 }
            ]
        }
