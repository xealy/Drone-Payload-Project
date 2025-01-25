# Drone-Payload-Project

![university project](https://img.shields.io/badge/university%20project-1E90FF)
![final year](https://img.shields.io/badge/final%20year-8A2BE2)

## About
The Drone Payload Project involves developing a payload for the X500 UAV to perform real-time air quality monitoring, target detection, and soil sampling in a simulated industrial environment. Key functions include identifying and evaluating the open/closed status of ball valves, an air pressure gauge, and ArUCOs marker, while integrating a soil sampling mechanism triggered by specific air pressure conditions.

## How to run
IMPORTANT: Ensure that in order to run any scripts, activate the 'egh455-merged-env' virtual environment
* to activate environment: source egh455-merged-env/bin/activate
* to deactivate environment: deactivate

### Web Visualisation Subsystem
To run the web viz app, in a separate terminal:
* first navigate to the directory that contains 'main.py'
* export FLASK_APP=main.py (if you use Mac)
* set FLASK_APP=main.py (if you use Windows -> you may also need to run $env:FLASK_APP="main.py")
* flask run --host=0.0.0.0
OR 
* python main.py

### Air Quality Subsystem
To run the air quality sensing script, find and run the 'aq-reading.py' file in the Repo in a separate terminal. This will start taking air quality readings and send the data as POST requests to the Flask App (make sure Flask App is running too).

## Concept of Operations
<img src="https://github.com/user-attachments/assets/e5febf42-d247-47a8-80db-e96a0040c40f" width="600" />

## Demo Videos
~ pending ~

