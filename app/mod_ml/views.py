from flask import Blueprint, render_template, request, redirect, url_for, session
from flask import jsonify
from flask import Response
import pandas as pd
import os
import json
import shutil
from datetime import datetime

ml_blueprint = Blueprint('ml', __name__)

from app.mod_eda.controllers import readFile, validFile
from app.mod_ml.controllers import runML

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPDIR = os.path.join(APP_ROOT, '../resources/temp/')
DEPLOYED = os.path.join(APP_ROOT, '../resources/deployed/')

@ml_blueprint.route('/')
def index():
	if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
	return redirect(url_for('auth.login'))

@ml_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
    
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['model_name'] = request.form.get('model_name')
        payload['target'] = request.form.getlist('target_single')
        #payload['targets'] = request.form.getlist('target')
        payload['contribute'] = request.form.get('contribute')
        payload['mode'] = request.form.get('mode')

        payload['drops'] = request.form.getlist('drop')

        payload['missing'] = request.form.get('missing')
        payload['encoding'] = request.form.get('encoding')
        payload['normalization'] = request.form.get('normalization')
        payload['dim_red'] = request.form.get('dim_red')
        payload['validation'] = request.form.get('validation')
        payload['fold'] = request.form.get('fold')
        payload['holdout'] = request.form.get('holdout')
        payload['tuning'] = request.form.get('tuning')
        payload['metrics'] = request.form.get('metrics')

        payload['best_config_file'] = 'best-config-classification.json'
        #payload['architecture_file'] = 'architecture-classification.json'
        #payload['weights_file'] = 'weights-classification.h5'

        #print(payload)

        #print('dt', request.form.get('todo_datetime'))
        #print('txt', request.form.get('todo_text'))

        results, cv, hy = runML(payload)

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        df.sort_values(cols[3], inplace=True, ascending=False)
        vals = df.values

        timestamp = datetime.now()
        owner = session['user']
        col_string = ','.join(map(str, cols.tolist()))
        val_string = ';'.join([','.join(map(str, val)) for val in vals])

        return render_template('ml/result.html', columns=cols, data=vals, cv=cv, hy=hy,
                                timestamp=timestamp, owner=owner,
                                col_string=col_string, val_string=val_string)
        
    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
    
    return redirect(url_for('auth.login'))