# ALEX MADE THIS, KIMBERLEY EDITS TO THRESHOLD gas and tune temperature 

import time
import datetime
from enviroplus import gas
from bme280 import BME280
from ltr559 import LTR559
from smbus2 import SMBus
# from pms5003 import PMS5003
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
# # Create ST7735 LCD display class
# st7735 = st7735.ST7735(
#     port=0,
#     cs=1,
#     dc="GPIO9",
#     backlight="GPIO12",
#     rotation=270,
#     spi_speed_hz=10000000
# )

# # Initialize display
# st7735.begin()
# WIDTH = st7735.width
# HEIGHT = st7735.height

# # Set up canvas and font
# img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
# draw = ImageDraw.Draw(img)
# path = os.path.dirname(os.path.realpath(__file__))
# font_size = 20
# font = ImageFont.truetype(UserFont, font_size)
# message = ""
# # The position of the top bar
# top_pos = 25

# # Create a values dict to store the data
# variables = ["temperature"]
# values = {}
# for v in variables:
#     values[v] = [1] * WIDTH


# # Displays data and text on the 0.96" LCD
# def display_text(variable, data, unit):
#     # Maintain length of list
#     values[variable] = values[variable][1:] + [data]
#     # Scale the values for the variable between 0 and 1
#     vmin = min(values[variable])
#     vmax = max(values[variable])
#     colours = [(v - vmin + 1) / (vmax - vmin + 1) for v in values[variable]]
#     # Format the variable name and value
#     message = f"{variable[:4]}: {data:.1f} {unit}"
#     draw.rectangle((0, 0, WIDTH, HEIGHT), (255, 255, 255))
#     for i in range(len(colours)):
#         # Convert the values to colours from red to blue
#         colour = (1.0 - colours[i]) * 0.6
#         r, g, b = [int(x * 255.0) for x in colorsys.hsv_to_rgb(colour, 1.0, 1.0)]
#         # Draw a 1-pixel wide rectangle of colour
#         draw.rectangle((i, top_pos, i + 1, HEIGHT), (r, g, b))
#         # Draw a line graph in black
#         line_y = HEIGHT - (top_pos + (colours[i] * (HEIGHT - top_pos))) + top_pos
#         draw.rectangle((i, line_y, i + 1, line_y + 1), (0, 0, 0))
#     # Write the text at the top in black
#     draw.text((0, 0), message, font=font, fill=(0, 0, 0))
#     st7735.display(img)

# Baseline resistance (R0) for gas types based on data sheet values 
R0_reducing = 100000 #(Ohms - 100kOhms) 
R0_oxidising = 20000 #(Ohms - 20 kOhms) 
R0_Nh3 = 150000 #(Ohms - 150 kOhms) 

co_a, co_b = 0.72, 1.4 # carbon monoxide values from data sheet 
no2_a, no2_b = 1.15, 0.02 #nitrogen dioxide values from data sheet 
nh3_a, nh3_b = 0.33, 0.0 #ammonia values from data sheet 

# Temperature tuning factor for compensation 
factor = 2.25 
cpu_temps = [] 

# calculate ppm 
def calculate_ppm(resistance, R0, a, b):
    #calculate using linear formula y = ax+b
    ratio = resistance / R0
    ppm = a*ratio + b
    return round(ppm, 2) 

# Function to get the temperature of the CPU
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = f.read()
            return int(temp) / 1000.0
    except FileNotFoundError:
        return 50.0  # Return a default value if file not found (e.g., on non-Raspberry Pi systems)


try:
    while True:
        # TIMESTAMP
        current_datetime = datetime.datetime.now()
        current_datetime_string = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
        print('-----------NEW READING AT (', current_datetime_string, ')-----------')

        # Get CPU temperature for compensation
        cpu_temp = get_cpu_temperature()
        cpu_temps.append(cpu_temp)
        
        if len(cpu_temps) > 5:  # Keep the last 5 CPU temperatures for averaging
            cpu_temps.pop(0)
        
        avg_cpu_temp = sum(cpu_temps) / len(cpu_temps)

        # WEATHER READINGS
        raw_temperature = round(bme280.get_temperature(), 2)
        # Compensate BME280 temperature using CPU temperature
        temperature = round(raw_temperature - ((avg_cpu_temp - raw_temperature) / factor))
        # temperature = round(bme280.get_temperature(), 2)
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

        # Compute PPM values for each gas type using the linear formula
        co_ppm = calculate_ppm(reducing, R0_reducing, co_a, co_b)       # Carbon Monoxide
        no2_ppm = calculate_ppm(oxidising, R0_oxidising, no2_a, no2_b)  # Nitrogen Dioxide
        nh3_ppm = calculate_ppm(nh3, R0_Nh3, nh3_a, nh3_b)              # Ammonia

        # OUTPUT
        print("Temperature: ", weather[0], "°C")
        print("Humidity: ", weather[1], "%")
        print("Pressure: ", weather[2])
        print("Light: ", weather[3], "Lux")
        #print("Reducing Gases: ", gases[0], "Ohms")
        #print("Oxidising Gases: ", gases[1], "Ohms")
        #print("NH3 Gases: ", gases[2], "Ohms")
        print("CO (PPM): ", co_ppm, "PPM")  # Display in PPM
        print("NO2 (PPM): ", no2_ppm, "PPM")  # Display in PPM
        print("NH3 (PPM): ", nh3_ppm, "PPM")  # Display in PPM 



        # TEMPERATURE ON LCD
        # unit = "°C"
        # data = temperature
        # display_text(variables[0], data, unit)

        # DEFINE POST REQUEST JSON
        data = {
            "timestamp": current_datetime_string,
            "temperature": weather[0],
            "humidity": weather[1],
            "pressure": weather[2],
            "light": weather[3],
            "reducing_gases": co_ppm,
            "oxidising_gases": no2_ppm,
            "nh3_gases": nh3_ppm
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
