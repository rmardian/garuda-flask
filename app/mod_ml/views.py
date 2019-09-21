from flask import Blueprint, render_template, request, redirect, url_for, session
from flask import jsonify
from flask import Response
import pandas as pd
import os
import json

ml_blueprint = Blueprint('ml', __name__)

from app.mod_eda.controllers import readFile, validFile
from app.mod_ml.controllers import runML

@ml_blueprint.route('/')
def index():
	if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
	return redirect(url_for('auth.login'))

'''
@ml_blueprint.route('/run_old', methods=['GET', 'POST'])
def run_old():

    if request.method == 'POST':
        
        payload = {}

        payload['contribute'] = request.form.get('contribute')

        payload['filename'] = request.form.get('filename')
        payload['mode'] = request.form.get('submit')
        payload['encoding'] = request.form.get('encoding')
        payload['missing'] = request.form.get('missing')
        payload['normalization'] = request.form.get('normalization')
        payload['targets'] = request.form.getlist('target')
        payload['crossval'] = request.form.get('crossval')
        payload['drops'] = request.form.getlist('drop')
        payload['test-size'] = request.form.get('test-size')
        payload['cv_method'] = request.form.get('cv_method')
        payload['dim_red'] = request.form.get('dim_red')
        payload['num_of_dim'] = request.form.get('dimension')
        payload['hyper-param'] = request.form.get('hyper-param')
        payload['tuning'] = request.form.get('tuning')
        payload['grids'] = request.form.get('grids')
        payload['model-name'] = request.form.get('model-name')

        payload['filter'] = 'regime'	#this value only matter for regression
        payload['selected_condition'] = 2	#Or 2, this value will not matter for regime classification

        payload['save-best-config'] = True
        payload['best-config-file'] = 'best-config-classification.json'
        payload['save-architecture'] = True
        payload['architecture-file'] = 'architecture-classification.json'
        payload['save-weights'] = True
        payload['weights-file'] = 'weights-classification.h5'

        payload['epoch'] = request.form.get('epoch')
        payload['batch'] = request.form.get('batch')
        payload['num_layers'] = request.form.get('num_layers')
        payload['num_nodes'] = request.form.get('num_nodes')

        ### this actually handled by javascript
        if payload['epoch'] != "" and payload['epoch'] is not None:
            epochs = list(map(int, payload['epoch'].split(',')))
        else:
            epochs = [32]
        if payload['batch'] != "" and payload['batch'] is not None:
            batch_size = list(map(int, payload['batch'].split(',')))
        else:
            batch_size = [100]
        if payload['num_layers'] != "" and payload['num_layers'] is not None:
            num_hidden = list(map(int, payload['num_layers'].split(',')))
        else:
            num_hidden = [8]
        if payload['num_nodes'] != "" and payload['num_nodes'] is not None:
            node_hidden = list(map(int, payload['num_nodes'].split(',')))
        else:
            node_hidden = [8]
        ###
        
        if not payload['hyper-param']:
            tuning_params = {
                'batch_size': batch_size[0],
                'epochs': epochs[0],
                'node_hidden': node_hidden[0],
                'num_hidden': num_hidden[0]
            }
        else:
            tuning_params = {'mod__batch_size': batch_size,
				'mod__epochs': epochs,
				'mod__node_hidden': node_hidden,
				'mod__num_hidden': num_hidden
            }

        #if (len(payload['targets']) == 0):
        #    return 'Please pick one or more targets!'


        num_folds = int(1/float(payload['test-size']))
        if request.form.get('submit') == 'Classification':
            if (len(payload['targets']) > 1):
                return 'Multi-label Classification is not supported. Please go back!'

            payload['metrics'] = request.form.getlist('cls_metrics')


        elif request.form.get('submit') == 'Regression':
            if (len(payload['targets']) > 1):
                return 'Multi-label Regression is not supported. Please go back!'

            payload['metrics'] = request.form.getlist('reg_metrics')

        results = runML(payload)

        cv = 'Yes' if payload['crossval'] or payload['hyper-param'] is not None else 'No'
        hy = 'Yes' if payload['hyper-param'] is not None else 'No'

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values

        print(results)

        return render_template('ml/result.html', columns=cols, data=vals, crossval=cv, hyperparam=hy)
        
    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
    
    return redirect(url_for('auth.login'))
'''

@ml_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
    
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['model_name'] = request.form.get('model_name')
        payload['target'] = request.form.getlist('target_single')
        payload['targets'] = [payload['target']]
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

        print(payload)

        #print('dt', request.form.get('todo_datetime'))
        #print('txt', request.form.get('todo_text'))

        if payload['mode'] == 'classification':

            #payload['metrics'] = request.form.getlist('cls_metrics')

            results = runML(payload)

            #df = pd.DataFrame(tuple_test, columns=df_test.columns)
            #df.sort_values('accuracy', ascending=False, inplace=True)
            #df.dropna(axis=1, inplace=True)
            #df = df.round(3)

        elif payload['mode'] == 'clustering':

            print('clustering')

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        df.sort_values(cols[3], inplace=True, ascending=False)
        vals = df.values

        return render_template('ml/result.html', columns=cols, data=vals)
        
        #return render_template('ml/result.html', columns=df.columns, data=df.values,
        #                        values1=df['accuracy'], labels=df['model_name'],
        #                        values2=df['precision'], test=test, preds=preds, probs=probs, roc_curves=roc_curves
        #                       )
    
    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
    
    return redirect(url_for('auth.login'))

'''
@ml_blueprint.route('/details', methods=['GET', 'POST'])
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

    if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))

    return redirect(url_for('auth.login'))

'''

