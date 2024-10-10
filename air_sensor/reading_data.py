import time 
import requests 
from datetime import datetime 
# from smbus2 import SMBus - may not be needed, or needed for temp 
from bme280 import BME280 
from ltr559 import LTR559 
from enviroplus import gas 
from pms5003 import PMS5003 
from enviroplus.noise import Noise 
from subprocess import PIPE, Popen
import numpy as np 

#date time stamp 
now = datetime.now()
#record in flie with everything and send all as a json packet for the webserver 

# CPU temperature set up
def get_cpu_temperature():
    process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE, universal_newlines=True)
    output, _error = process.communicate()
    return float(output[output.index('=') + 1:output.rindex("'")])

# temperature set up 
def get_temperature():
    cpu_temp=get_cpu_temperature()
    cpu_temps = [get_cpu_temperature()]*5 
    cpu_temps = cpu_temps[1:] + [cpu_temp]
    avg_cpu_temp = sum(cpu_temps) / float(len(cpu_temps))
    raw_temp = bme280.get_temperature()
    temperature = raw_temp - ((avg_cpu_temp - raw_temp) / factor)
    return temperature

# gas sensor 
def get_gases():
    readings = gas.read_all() 
    reduced = round(readings.reducing, 2) # readings.'reducing' will just give reducing change '' to get parts needed 
    oxidised = round(readings.oxidising, 2)
    nh3 = round(readings.nh3, 2)
    gases = [oxidised, reduced,nh3]
    return gases 

start_time = time.time() 
gases_data = get_gases() 
#put gas data in file 
file1 = open("all_readings.txt", "w")
file1.seek(0)
file1.truncate() 

try:
    while True:

        elapsed_time = time.time() - start_time
        
        if elapsed_time<120:
            gases_threshold=get_gases()
            pressure_calibration=bme280.get_pressure()
            humidity_calibration=bme280.get_humidity() #humidity set up 
            light_calibration=ltr559.get_lux() #light set up 
            temperature_calibration= get_temperature()
	    

        elif elapsed_time>120:

            run_time=elapsed_time -120
            
            pressure=bme280.get_pressure()
            print("pressure=", pressure)

            humidity=bme280.get_humidity()
            print("humidity=",humidity)

            light=ltr559.get_lux()
            print("light=",light)

            temperature= get_temperature()
            print("temperature=",temperature)

            gases=get_gases()

            val=[str(run_time),str(pressure),str(humidity),str(light),str(temperature),str(gases_threshold),str(gases)]

            val_str = ",".join(val)
            file1.write(val_str + "\n") 

            if gases[0]>gases_threshold[0]:
                print("OX current value: ",gases[0]," OX threshold: ",gases_threshold[0], " OX concentration is increasing")
            elif gases[0]<gases_threshold[0]:
                print("OX current value: ",gases[0]," OX threshold: ",gases_threshold[0], " OX concentration is reducing")
            else:
                print("OX current value: ",gases[0]," OX threshold: ",gases_threshold[0], " OX concentration is the same as threshold")

            if gases[1]>gases_threshold[1]:
                print("RED current value: ",gases[1]," RED threshold: ",gases_threshold[1], " RED concentration is reducing")
            elif gases[1]<gases_threshold[1]:
                print("RED current value: ",gases[1]," RED threshold: ",gases_threshold[1], " RED concentration is increasing")
            else:
                print("RED current value: ",gases[1]," RED threshold: ",gases_threshold[1], " RED concentration is the same as threshold")

            if gases[2]>gases_threshold[2]:
                print("NH3 current value: ",gases[2]," NH3 threshold: ",gases_threshold[2], " NH3 concentration is reducing")
            elif gases[2]<gases_threshold[2]:
                print("NH3 current value: ",gases[2]," NH3 threshold: ",gases_threshold[2], " NH3 concentration is increasing")
            else:
                print("NH3 current value: ",gases[2]," NH3 threshold: ",gases_threshold[2], " NH3 concentration is the same")

            time.sleep(5)
            
except KeyboardInterrupt:
    file1.close()
    pass