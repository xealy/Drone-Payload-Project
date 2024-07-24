# EGH455-UAV-Project
Repository for EGH455 UAV Project

## Dashboard web app
To run the web app, I suggest you use a virtual environment to keep package installations confined to an environment. For this, first install Anaconda and follow the next steps.

### To set up conda environment:
conda create --name egh455-ui python=3.11
conda activate egh455-ui
pip install -r requirements.txt
pip install pychart.js (run this too because i didn't include in requirements.txt)

### To run flask app:
export FLASK_APP=main.py (if you use Mac)
$env:FLASK_APP="main.py" (if you use Windows)
flask run
