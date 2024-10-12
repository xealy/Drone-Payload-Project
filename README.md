# EGH455-UAV-Project
Repository for EGH455 UAV Project

IMPORTANT: Ensure that in order to run any scripts, activate the 'egh455-merged-env' virtual environment
* to activate environment: source egh455-merged-env/bin/activate
* to deactivate environment: deactivate

## WEB VIZ
To run the web viz app, in a separate terminal:
* first navigate to the directory that contains 'main.py'
* export FLASK_APP=main.py (if you use Mac)
* set FLASK_APP=main.py (if you use Windows -> you may also need to run $env:FLASK_APP="main.py")
* flask run --host=0.0.0.0
OR 
* python main.py

## AIR QUALITY
To run the air quality sensing script, find and run the 'aq-reading.py' file in the Repo in a separate terminal. This will start taking air quality readings and send the data as POST requests to the Flask App (make sure Flask App is running too).


