import logging
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    'host': 'localhost',
    'port': '27017',
    'database': 'aegisdb_staging_2',
    'username': 'root',
    'password': 'root'
}
'''
SOURCE = {
    'host': '130.211.200.48',
    'port': '3306',
    'database': 'qluein',
    'username': 'root',
    'password': 'qlue7654'
}
'''
SOURCE = {
    'host': 'localhost',
    'port': '3306',
    'database': 'sakila',
    'username': 'root',
    'password': ''
}

ALGORITHMS = {
    'supervised': ['SVC', 'LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GaussianNB'],
    'unsupervised': ['KMeans', 'GaussianMixture'],
    'regression': ['LinearRegression', 'RandomForestRegressor'],
    'dimension-reduction': ['PCA']
}

class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'd44e9839f0d9e80cd941b9adaa633033eed9659460e53dd0'
    #CSRF_ENABLED = True
    #SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']

class ProductionConfig(Config):
    DEBUG = False

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s ' \
                 '[in %(pathname)s:%(lineno)d]'
    LOGGING_LOCATION = os.path.join(PROJECT_ROOT, 'instance', 'app.log')
    LOGGING_LEVEL = logging.DEBUG

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class TestingConfig(Config):
    TESTING = True