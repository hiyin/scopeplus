# -*- coding: utf-8 -*-

from flask import Flask

from .config import DefaultConfig
from .user import Users, UsersAdmin
from .settings import settings
from .tasks import tasks
from .naso_tableview import naso
from .frontend import frontend, ContactUsAdmin
from .extensions import db, mail, cache, login_manager, admin, mongo
from .utils import pretty_date
import logging
import os
from datetime import timedelta

from pymongo import MongoClient
from .model.umap import UmapView
from .model.meta import MetaView

# For import *
__all__ = ['create_app']

DEFAULT_BLUEPRINTS = (
    frontend,
    settings,
    tasks,
    naso
)

logging.basicConfig(level=logging.DEBUG,
                   format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                   datefmt='%Y-%m-%d %H:%M:%S',
                   handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def create_app(config=None, app_name=None, blueprints=None):
    # Create a Flask app

    if app_name is None:
        app_name = DefaultConfig.PROJECT
    if blueprints is None:
        blueprints = DEFAULT_BLUEPRINTS

    app = Flask(app_name,
                instance_relative_config=True)

    # Add 0627 by junyi
    # app.config['LOGIN_DISABLED'] = True
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

    # add db


    configure_app(app, config)
    configure_hook(app)
    configure_blueprints(app, blueprints)
    configure_extensions(app)
    configure_logging(app)
    configure_template_filters(app)
    configure_error_handlers(app)

    return app


def configure_app(app, config=None):
    # Different ways of configurations i.e local or production
    logger.info(f"Starting app in %s environment" % os.getenv('FLASK_ENV'))
    flask_env = os.environ.get('FLASK_ENV')
    if 'development' in flask_env:
        print('configuring for development')
        app.config.from_object(DefaultConfig)
    # if exists under root ./instance/X.cfg
    # app.config.from_pyfile('secret_config.cfg')
    # print(app.instance_path)
    if config:
        app.config.from_object(config)


def configure_extensions(app):

    # flask-sqlalchemy
    db.init_app(app)

    # flask-mail
    mail.init_app(app)

    # flask-cache
    cache.init_app(app)

    # flask-admin
    # sqlite db data
    admin.add_view(ContactUsAdmin(db.session))
    admin.add_view(UsersAdmin(db.session))

    # mongo db data
    admin.add_view(UmapView(mongo['umap']))
    admin.add_view(MetaView(mongo['single_cell_meta_country']))

    admin.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return Users.query.get(id)
    login_manager.setup_app(app)


def configure_blueprints(app, blueprints):
    # Configure blueprints in views

    for blueprint in blueprints:
        app.register_blueprint(blueprint)


def configure_template_filters(app):

    @app.template_filter()
    def _pretty_date(value):
        return pretty_date(value)

    @app.template_filter()
    def format_date(value, format='%Y-%m-%d'):
        return value.strftime(format)


def configure_logging(app):
    # Configure logging

    if app.debug:
        # Skip debug and test mode. Better check terminal output.
        return

    # TODO: production loggers for (info, email, etc)


def configure_hook(app):
    @app.before_request
    def before_request():
        pass


def configure_error_handlers(app):

    @app.errorhandler(403)
    def forbidden_page(error):
        return "Oops! You don't have permission to access this page.", 403

    @app.errorhandler(404)
    def page_not_found(error):
        return "Opps! Page not found.", 404

    @app.errorhandler(500)
    def server_error_page(error):
        return "Oops! Internal server error. Please try after sometime.", 500

