# -*- coding: utf-8 -*-

import os
from .utils import TMP_FOLDER
class BaseConfig(object):
    # Change these settings as per your needs

    PROJECT = "flaskstarter"
    PROJECT_NAME = "flaskstarter.domain"
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    BASE_URL = "https://yourdomain-flaskstarter.domain"
    ADMIN_EMAILS = ['admin@flaskstarter.domain']

    DEBUG = False
    TESTING = False

    SECRET_KEY = 'always-change-this-secret-key-with-random-alpha-nums'
    HOST = '0.0.0.0'


class DefaultConfig(BaseConfig):
    FLASK_ENV = 'development'
    DEBUG = True

    # Flask-Sqlalchemy
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # SQLITE for dev
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + TMP_FOLDER + '/db.sqlite'



    # POSTGRESQL for dev
    # SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:pass@ip/dbname'

    # Flask-cache
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 60

    # Flask-mail
    MAIL_DEBUG = False
    MAIL_SERVER = ""  # something like 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True

    # Keep these in instance folder or in env variables
    MAIL_USERNAME = "admin-mail@yourdomain-flaskstarter.domain"
    MAIL_PASSWORD = ""
    MAIL_DEFAULT_SENDER = MAIL_USERNAME



