# ALEX MADE THIS

import time
import datetime
from enviroplus import gas
from bme280 import BME280
from ltr559 import LTR559
from smbus2 import SMBus
import requests

try:
    while True:
        # TIMESTAMP
        current_datetime = datetime.datetime.now()
        current_datetime_string = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
        print('-----------NEW READING AT (', current_datetime_string, ')-----------')

        # INITIALISE SENSORS
        bus = SMBus(1)
        bme280 = BME280(i2c_dev=bus)
        ltr559 = LTR559()

        # WEATHER READINGS
        temperature = round(bme280.get_temperature(), 2)
        humidity = round(bme280.get_humidity(), 2)
        pressure = round(bme280.get_pressure(), 2)
        light = round(ltr559.get_lux(), 2)
        weather = [temperature, humidity, pressure, light]

        # GAS READINGS
        readings = gas.read_all()
        reducing = round(readings.reducing, 2) # readings.'reducing' will just give reducing change '' to get parts needed 
        oxidising = round(readings.oxidising, 2)
        nh3 = round(readings.nh3, 2)
        gases = [reducing, oxidising, nh3]

        # OUTPUT
        print("Temperature: ", weather[0], "°C")
        print("Humidity: ", weather[1], "%")
        print("Pressure: ", weather[2])
        print("Light: ", weather[3], "Lux")
        print("Reducing Gases: ", gases[0], "Ohms")
        print("Oxidising Gases: ", gases[1], "Ohms")
        print("NH3 Gases: ", gases[2], "Ohms")

        # POST REQUEST
        data = {
            "timestamp": current_datetime_string,
            "temperature": weather[0],
            "humidity": weather[1],
            "pressure": weather[2],
            "light": weather[3],
            "reducing_gases": gases[0],
            "oxidising_gases": gases[1],
            "nh3_gases": gases[2]
        }

        try:
            response = requests.post("http://127.0.0.1:5000/", json=data)
            if response.status_code == 200:
                print("Data posted successfully.")
            else:
                print(f"Failed to post data. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error posting data: {e}")


        time.sleep(4.0)
except KeyboardInterrupt:
    pass
