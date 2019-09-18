from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename

nn_blueprint = Blueprint('nn', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_NN.controllers import validFile, getDataType, runNN

@nn_blueprint.route('/')
@nn_blueprint.route('/index')
@nn_blueprint.route('/index.html')
def index():

    return render_template('dafd/index.html')

@nn_blueprint.route('/analysis', methods=['GET', 'POST'])
def analysis():

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
        
        return render_template('dafd/analysis.html', columns=columns, data=df.values, filename=filename)

    return redirect(url_for('nn.index'))

@nn_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
        
        compare = False
        
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['mode'] = request.form.get('submit')
        payload['encoding'] = request.form.get('encoding')
        payload['missing'] = request.form.get('missing')
        payload['normalization'] = request.form.get('normalization')
        payload['targets'] = request.form.getlist('target')
        payload['drops'] = request.form.getlist('drop')
        payload['crossval'] = request.form.get('crossval')
        payload['test-size'] = request.form.get('test-size')
        payload['cv_method'] = request.form.get('cv_method')
        payload['dim_red'] = request.form.get('dim_red')
        payload['num_of_dim'] = request.form.get('dimension')
        payload['hyper-param'] = request.form.get('hyper-param')
        payload['tuning'] = request.form.get('tuning')
        payload['grids'] = request.form.get('grids')

        #if (len(payload['targets']) == 0):
        #    return 'Please pick one or more targets!'

        if request.form.get('submit') == 'Classification':
            if (len(payload['targets']) > 1):
                return 'Multi-label Classification'

            payload['metrics'] = request.form.getlist('metrics')

            columns, tuples, cv_results = runNN(payload, compare)
            num_folds = int(1/float(payload['test-size']))
            cv_columns = list(range(len(cv_results['split0_test_score'])))

            print(cv_results['split0_test_score'])

            df = pd.DataFrame(tuples, columns=columns)
            df.sort_values('accuracy', ascending=False, inplace=True)
            df.dropna(axis=1, inplace=True)
            df = df.round(3)

        elif request.form.get('submit') == 'Regression':
            if (len(payload['targets']) > 1):
                return 'Multi-label Regression'

            payload['metrics'] = request.form.getlist('reg_metrics')

            return 'Regression'

        cv = 'No'
        if payload['crossval'] is not None:
            cv = 'Yes'
        
        return render_template('dafd/result.html', columns=df.columns, data=df.values,
                                values1=df['accuracy'], labels=df['model_name'], crossval=cv,
                                cv_results=cv_results, num_folds=num_folds, cv_columns=cv_columns)

    return redirect(url_for('nn.index'))



