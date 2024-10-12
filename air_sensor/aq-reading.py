import time
import datetime
from enviroplus import gas
from bme280 import BME280
from ltr559 import LTR559
from smbus2 import SMBus
# from pms5003 import PMS5003
import requests

# INITIALISE SENSORS
bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)
ltr559 = LTR559()

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
    # calculate using linear formula y = ax+b
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


# FOR PROXIMITY SWITCH
delay = 0.5  # Debounce the proximity tap
mode = 0  # The starting mode
last_page = 0
light = 1

try:
    while True:
        # PROXIMITY SWITCH
        change_lcd = 'False'
        proximity = ltr559.get_proximity()

        # If the proximity crosses the threshold, toggle the mode
        if proximity > 1500 and time.time() - last_page > delay:
            change_lcd = 'True'
            last_page = time.time()

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
        print("CO (PPM): ", co_ppm, "PPM")  # Display in PPM
        print("NO2 (PPM): ", no2_ppm, "PPM")  # Display in PPM
        print("NH3 (PPM): ", nh3_ppm, "PPM")  # Display in PPM 
        print("CHANGE LCD BOOL: ", change_lcd)  # Display in PPM 

        # DEFINE POST REQUEST JSON
        data = {
            "timestamp": current_datetime_string,
            "temperature": weather[0],
            "humidity": weather[1],
            "pressure": weather[2],
            "light": weather[3],
            "reducing_gases": co_ppm,
            "oxidising_gases": no2_ppm,
            "nh3_gases": nh3_ppm, 
            "change_lcd": change_lcd
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
