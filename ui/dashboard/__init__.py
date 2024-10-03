import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# function that creates a web app
def create_app():
    app = Flask(__name__)
    app.debug = True
    app.secret_key = 'utroutoru'

    # configure sqlalchemy database uri in app
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    # importing views module -> practice to prevent circular references
    from . import views
    app.register_blueprint(views.bp)

    return app
