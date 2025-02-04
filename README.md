# Drone Payload Project

## About
The Drone Payload Project involves developing a payload for a UAV to perform real-time air quality monitoring, target detection, and soil sampling in a simulated industrial environment. Key functions include identifying and evaluating the open/closed status of ball valves, an air pressure gauge, and ArUCOs marker, while integrating a soil sampling mechanism triggered by specific air pressure conditions.

## How to run
* create new virtual environment with the requirements from 'requirements-MERGED-FINAL-W-SS.txt'
* to activate environment: source your-env-here/bin/activate
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
### Air Quality + Logging Demo
https://github.com/user-attachments/assets/765d6303-6741-42a4-9d52-9b161f32e4a7

### Target Acquisition Demo
https://github.com/user-attachments/assets/a84afd88-2d94-4411-9111-2591d38376b9

### LCD Demo
https://github.com/user-attachments/assets/2332168d-ff42-44fd-9390-0a332bcdd576


