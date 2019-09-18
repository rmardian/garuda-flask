from flask import Blueprint, render_template, request, redirect, url_for, session
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename

eda_blueprint = Blueprint('eda', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_eda.controllers import runML, validFile, getDataType, getROCparams

@eda_blueprint.route('/')
@eda_blueprint.route('/index')
@eda_blueprint.route('/index.html')
def index():
    return render_template('eda/index.html')

@eda_blueprint.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        
        file = request.files['file']

        if not file:
            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = 'dafd.csv'
            complete_filename = os.path.join(target, filename)
        
        if validFile(file.filename):

            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            complete_filename = os.path.join(target, secure_filename(filename))
            file.save(complete_filename)

        df = getDataType(filename)
        df = df.round(3)
        columns = df.columns.tolist()
        
        return render_template('eda/analysis.html', columns=columns, data=df.values, filename=filename)

    return redirect(url_for('eda.index'))

@eda_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':

        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['mode'] = request.form.get('submit')

        payload['missing'] = request.form.get('missing')
        payload['normalization'] = request.form.get('normalization')

        #print('dt', request.form.get('todo_datetime'))
        #print('txt', request.form.get('todo_text'))

        if request.form.get('submit') == 'supervised':

            payload['targets'] = request.form.getlist('target')
            payload['drops'] = request.form.getlist('drop')
            payload['crossval'] = request.form.get('crossval')
            payload['test-size'] = request.form.get('test-size')
            payload['cv_method'] = request.form.get('cv_method')
            payload['dim_red'] = request.form.get('dim_red')
            payload['num_of_dim'] = request.form.get('dimension')
            payload['grids'] = request.form.get('grids')
            payload['metrics'] = request.form.getlist('metrics')

            df_test, test, preds, probs, tuple_test, roc_curves = runML(payload)

            df = pd.DataFrame(tuple_test, columns=df_test.columns)
            df.sort_values('accuracy', ascending=False, inplace=True)
            df.dropna(axis=1, inplace=True)
            df = df.round(3)

        elif request.form.get('submit') == 'unsupervised':

            print('unsupervised')
        
        print(roc_curves[0][0])
        return render_template('eda/result.html', columns=df.columns, data=df.values,
                                values1=df['accuracy'], labels=df['model_name'],
                                values2=df['precision'], test=test, preds=preds, probs=probs, roc_curves=roc_curves
                                )

    return redirect(url_for('eda.index'))

@eda_blueprint.route('/details', methods=['GET', 'POST'])
def details():

    if request.method == 'POST':

        list_of_model = request.form.getlist('model')
        test = request.form.get('test')
        #print(list_of_model)
        #print(request.form.get('preds'))
        probs = request.form.get('probs')

        print(session['test'])

        #print(json.loads(probs))
        #print(probs[0])
        for i in range(0, len(list_of_model)):
            if request.form.get('details_' + str(i+1)) != None:
                #print(list_of_model[i])
                #print(test)
                #print(probs[i])
                #getROCparams(session['test'], probs[0])
                #return 'OK'
                return render_template('eda/details.html', name=list_of_model[i])

    return redirect(url_for('eda.index'))
