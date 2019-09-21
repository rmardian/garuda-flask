from flask import Blueprint, render_template, request, redirect, url_for, session
import os
from time import time
from werkzeug.utils import secure_filename

eda_blueprint = Blueprint('eda', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_eda.controllers import validFile, getDataType

@eda_blueprint.route('/')
def index():
	if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
	return redirect(url_for('auth.login'))

@eda_blueprint.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        
        file = request.files['file']
        '''
        if not file:
            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = 'dafd.csv'
            complete_filename = os.path.join(target, filename)
        '''
        
        if validFile(file.filename):

            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            path_to_file = os.path.join(target, secure_filename(filename))
            file.save(path_to_file)

        df = getDataType(path_to_file)
        df = df.round(3)
        columns = df.columns.tolist()
        
        model_name = 'md-' + str(round(time() * 1000))
        
        return render_template('eda/analysis.html', columns=columns, data=df.values, filename=filename,
                                    model_name=model_name)

    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))

    return redirect(url_for('auth.login'))

#DUMMY ENDPOINTS for testing purpose
@eda_blueprint.route('/dummy', methods=['GET'])
def dummy():

    target = os.path.join(APP_ROOT, '../resources/inputs/')
    filename = 'dafd.csv'
    complete_filename = os.path.join(target, filename)

    df = getDataType(filename)
    df = df.round(3)
    columns = df.columns.tolist()
    
    model_name = 'md-' + str(round(time() * 1000))
    
    return render_template('eda/analysis.html', columns=columns, data=df.values, filename=filename,
                                model_name=model_name)


