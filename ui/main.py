from dashboard import create_app
from flask import Flask
from dashboard.views import bp

app = Flask(__name__)

app = create_app()
app.register_blueprint(bp)

# @app.route('/')
# def home():
#     return "Hello, World!"

if __name__ == '__main__':
    # app = create_app()
    app.run(debug=False, host='0.0.0.0')
