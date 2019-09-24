from flask import Blueprint, render_template, request, redirect, url_for, session, Response, send_from_directory
from flask import jsonify
from flask import Response
import pandas as pd
import os
import json
import csv
import shutil
from datetime import datetime
from time import time
from werkzeug.utils import secure_filename

predict_blueprint = Blueprint('predict', __name__)

from app.mod_predict.controllers import validFile, runPrediction

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')
TEMPDIR = os.path.join(APP_ROOT, '../resources/temp/')
DEPLOYED = os.path.join(APP_ROOT, '../resources/deployed/')

@predict_blueprint.route('/')
def index():
	if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
	return redirect(url_for('auth.login'))

@predict_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
    
        file = request.files['file']
        if validFile(file.filename):
            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            path_to_file = os.path.join(target, secure_filename(filename))
            file.save(path_to_file)

        modelName = request.form.get('model')
        output = runPrediction(path_to_file, modelName)
        col_string = ','.join(map(str, output))
        output_name = 'pred-' + str(round(time() * 1000)) + '.csv'

        return render_template('predict/result.html', prediction=col_string,
                                filename=filename, output_name=output_name)
        
    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
    
    return redirect(url_for('auth.login'))


@predict_blueprint.route('/download', methods=['GET', 'POST'])
def download():

    if request.method == 'POST':

        prediction = request.form.get('prediction')
        output_name = request.form.get('output_name')
        filename = request.form.get('filename')
        path_to_file = os.path.join(RESOURCES, filename)

        csv = ''
        if request.form.get('download') == 'Download with Full Dataset':
            idx = 0
            predlist = prediction.split(',')
            with open(path_to_file) as contents:
                for row in contents:
                    row = row.rstrip('\n')
                    if idx == 0:
                        csv += row + ',' + 'Predicted Class\n'
                    else:
                        csv += row + ',' + predlist[idx-1] + '\n'
                    idx += 1
        
        else:
            csv = 'Predicted Class\n'
            for i in prediction.split(','):
                csv += (i + '\n')

        return Response(csv, mimetype='text/csv',
                        headers={'Content-disposition': 'attachment; filename=' + output_name})
        
    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
    
    return redirect(url_for('auth.login'))