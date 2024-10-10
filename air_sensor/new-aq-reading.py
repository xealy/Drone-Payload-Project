# ALEX MADE THIS

import time
import datetime
from enviroplus import gas
from bme280 import BME280
from ltr559 import LTR559
from smbus2 import SMBus
import requests

# LCD IMPORTS
import colorsys
import os
import sys
import st7735
from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont

# INITIALISE SENSORS
bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)
ltr559 = LTR559()

# INITIALISE LCD
# Create ST7735 LCD display class
st7735 = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=10000000
)

# Initialize display
st7735.begin()
WIDTH = st7735.width
HEIGHT = st7735.height
# Set up canvas and font
img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
draw = ImageDraw.Draw(img)
path = os.path.dirname(os.path.realpath(__file__))
font_size = 20
font = ImageFont.truetype(UserFont, font_size)
message = ""
# The position of the top bar
top_pos = 25


# Create a values dict to store the data
variables = ["temperature"]
values = {}
for v in variables:
    values[v] = [1] * WIDTH


# Displays data and text on the 0.96" LCD
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
    st7735.display(img)


try:
    while True:
        # TIMESTAMP
        current_datetime = datetime.datetime.now()
        current_datetime_string = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
        print('-----------NEW READING AT (', current_datetime_string, ')-----------')

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

        # TEMPERATURE ON LCD
        unit = "°C"
        data = temperature
        display_text(variables[0], data, unit)

        # DEFINE POST REQUEST JSON
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

        # SEND POST REQUEST
        try:
            response = requests.post("http://127.0.0.1:5000/", json=data)
            if response.status_code == 200:
                print("Data posted successfully.")
            else:
                print(f"Failed to post data. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error posting data: {e}")

        # SLEEP for 4 seconds
        time.sleep(4.0)

except KeyboardInterrupt:
    pass
