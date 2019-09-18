from flask import Blueprint, render_template, request, redirect, url_for
from flask import jsonify
from flask import Response
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename

ml_blueprint = Blueprint('ml', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_eda.controllers import validFile, getDataType

@ml_blueprint.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        if request.is_json:

            print('get here')
            content = request.get_json()

            #js = json.dumps(content)
            #resp = Response(js, status=200, mimetype='application/json')

            resp = jsonify(content)
            resp.status_code = 200

            return resp
        
        return 'ITS FINE'

    return 'ML GET!'

@ml_blueprint.route('/ingest', methods=['GET', 'POST'])
def ingest():

    if request.method == 'POST':
        return getDataType('example.csv')

    return 'Ingested at EDA!'

