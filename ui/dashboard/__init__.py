import os
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# function that creates a web app
def create_app():
    app = Flask(__name__)
    app.debug = True
    app.secret_key = 'utroutoru'

    # configure sqlalchemy database uri in app
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

    if 'DATABASE_URL' in os.environ:
        app.config.from_mapping(
            SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
        )

    # initialise database
    db.init_app(app)
    Bootstrap(app)

    # importing views module -> practice to prevent circular references
    from . import views
    app.register_blueprint(views.bp)

    return app
