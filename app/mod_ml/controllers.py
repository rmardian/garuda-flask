from flask import session
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta

#from sklearn.externals import joblib 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from app.mod_ml.model import MODELS, PARAMS

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')

def readFile(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

def impute(df, missing):

	if missing == 'drop':
		df.dropna(inplace=True)

	else:
		imputer = Imputer(missing_values='NaN', strategy=missing, axis=0)
		imputer = imputer.fit(df)
		df = imputer.transform(df)

	return df

def scaling(X, normalization):

	if normalization == 'minmax':
		X_scaled = MinMaxScaler().fit_transform(X)
	elif normalization == 'standard':
		X_scaled = StandardScaler().fit_transform(X)
	elif normalization == 'robust':
		X_scaled = RobustScaler().fit_transform(X)
	else:
		X_scaled = X

	return X_scaled

def getScore(name, scores, test, pred, prob):

	acc, prec, rec, f1, roc, mcc = None, None, None, None, None, None

	for score in scores:
		if score == 'acc':
			acc = accuracy_score(test, pred)
		elif score == 'precision':
			prec = precision_score(test, pred)
		elif score == 'recall':
			rec = recall_score(test, pred)
		elif score == 'f1':
			f1 = f1_score(test, pred)
		elif score == 'roc':
			roc = roc_auc_score(test, prob)
		elif score == 'mcc':
			mcc = matthews_corrcoef(test, pred)

	return name, acc, prec, rec, f1, roc, mcc

def getROCparams(test, prob):

	print(prob)
	#fpr, tpr, _ = roc_curve(test, prob)
	#return fpr, tpr


def makeDataset(df, targets):

	X = df.drop(targets, axis=1)
	y = df[targets]

	return X.values, y.values

def generateParams(payload):

	params = []
	if payload['missing'] != 'drop':
		imputer = SimpleImputer(missing_values='NaN', strategy=payload['missing'])
		params.append(('imputer', imputer))
	if payload['encoding'] != 'none':
		if payload['encoding'] == 'label':
			encoder = LabelEncoder()
		elif payload['encoding'] == 'onehot':
			encoder = OneHotEncoder()
		elif payload['encoding'] == 'binary':
			encoder = LabelEncoder()			#create binary encoder
		elif payload['encoding'] == 'count':
			encoder = LabelEncoder()			#create count encoder
		params.append(('encoder', encoder))
	if payload['normalization'] != 'none':
		if payload['normalization'] == 'minmax':
			scaler = MinMaxScaler()
		elif payload['normalization'] == 'standard':
			scaler = StandardScaler()
		elif payload['normalization'] == 'robust':
			scaler = RobustScaler()
		params.append(('scaler', scaler))
	if payload['dim_red'] != '':
		params.append(('reducer', PCA(n_components=int(payload['dim_red']))))

	return params

def getClassificationScore(name, scores, test, pred, prob):

	acc, prec, rec, f1, roc = None, None, None, None, None
	'''
	for score in scores:
		if score == 'accuracy':
			acc = accuracy_score(test, pred)
		elif score == 'precision':
			prec = precision_score(test, pred)
		elif score == 'recall':
			rec = recall_score(test, pred)
		elif score == 'f1':
			f1 = f1_score(test, pred)
		elif score == 'roc_auc':
			roc = roc_auc_score(test, prob)
	'''
	if scores == 'accuracy':
		return accuracy_score(test, pred)
	elif scores == 'precision':
		return precision_score(test, pred)
	elif scores == 'recall':
		return recall_score(test, pred)
	elif scores == 'f1':
		return f1_score(test, pred)
	elif scores == 'roc_auc':
		return roc_auc_score(test, prob)

	score_dict = {
		'Mode': 'Classification',
		'Model Name': name,
		'Accuracy': acc,
		'Precision': prec,
		'Recall': rec,
		'F-Score': f1,
		'ROC-AUC': roc
	}
	return {k:[v] for k,v in score_dict.items() if v is not None}

def getRegressionScore(name, scores, pred, test):

	mae, mse, r2 = None, None, None

	'''
	for score in scores:
		if score == 'mae':
			mae = mean_absolute_error(test, pred)
		elif score == 'mse':
			mse = mean_squared_error(test, pred)
		elif score == 'rmse':
			rmse = np.sqrt(mean_squared_error(test, pred))
		elif score == 'r2':
			r2 = r2_score(test, pred)
	'''

	if scores == 'neg_mean_absolute_error':
		return mean_absolute_error(test, pred)
	elif scores == 'neg_mean_squared_error':
		return mean_squared_error(test, pred)
	elif scores == 'r2':
		return r2_score(test, pred)

	score_dict = {
		'Mode': 'Regression',
		'Model Name': name,
		'Mean Absolute Error': mae,
		'Mean Squared Error': mse,
		'R-squared': r2
	} 
	return {k:[v] for k,v in score_dict.items() if v is not None}

def runML(payload):

	path_to_file = os.path.join(RESOURCES, payload['filename'])
	df = readFile(path_to_file)

	df.drop(payload['drops'], axis=1, inplace=True)		#drop first, so NaN could be minimized
	if payload['missing'] == 'drop':
		df.dropna(inplace=True)
	X, y = makeDataset(df, payload['target'])
	y = y.ravel()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['holdout']))

	results = {}
	results['Model Name'] = []
	results['Mode'] = []
	results['Algorithm'] = []
	results[payload['metrics'].upper()] = []

	for i in range(len(MODELS)):

		results['Model Name'].append(payload['model_name'] + '-' + str(i))
		results['Mode'].append(payload['mode'].upper())
		results['Algorithm'].append(MODELS[i][0])

		model = MODELS[i][1]
		idx = 'model' + str(i)

		params = generateParams(payload)
		params.append((idx, model))
		pipeline = Pipeline(params)
		exe = None
		result = 0.0

		if payload['tuning'] == 'grid':
			exe = GridSearchCV(pipeline, PARAMS[i], cv=int(payload['fold']), n_jobs=-1, verbose=2)
		elif payload['tuning'] == 'random':
			exe = RandomizedSearchCV(pipeline, PARAMS[i], cv=int(payload['fold']), n_jobs=-1, verbose=2)
		elif payload['tuning'] == 'bayesian':
			exe = GridSearchCV(pipeline, PARAMS[i], cv=int(payload['fold']), n_jobs=-1, verbose=2)	#change into hyperopt

		elif payload['validation'] == 'crossval':
			start = datetime.now()
			res = cross_validate(estimator=pipeline, X=X_train, y=y_train, cv=int(payload['fold']), scoring=payload['metrics'])
			end = datetime.now()
			print('Total execution time:', str(end-start))

			res_dict = {}
			res_dict['Model Name'] = payload['model_name']
			for s in payload['metrics']:
				key = 'test_' + s
				res_dict[key] = res[key].mean()

			return res_dict

		else:
			exe = pipeline
			print(pipeline)

		start = datetime.now()
		exe.fit(X_train, y_train)
		end = datetime.now()
		print('Total execution time:', str(end-start))

		if payload['tuning'] != 'none':

			top3 = pd.DataFrame(exe.cv_results_)
			top3.sort_values(by='rank_test_score', inplace=True)
			print(top3)

			best_params = exe.best_params_
			print('Best config:', best_params)
			y_pred = exe.best_estimator_.predict(X_test)
			
			if payload['contribute'] == 'agree':
				json_str = json.dumps(exe.best_params_)
				best_config_json = os.path.join(RESOURCES, payload['best_config_file'])
				with open(best_config_json, 'w') as json_file:
					json_file.write(json_str)

			if payload['mode']=='classification':
				y_prob = exe.best_estimator_.predict_proba(X_test)[:, 1]
				
			model_saver = exe.best_estimator_[idx]	#in keras model = exe.best_estimator_['model'].model

		else:
			y_pred = exe.predict(X_test)
			if payload['mode']=='classification':
				y_prob = exe.predict_proba(X_test)[:, 1]

			model_saver = pipeline.named_steps[idx]

		'''
		if payload['contribute'] == 'agree':
		
			architecture = os.path.join(RESOURCES, payload['architecture_file'])
			with open(architecture, 'w') as json_file:
				json_file.write(model_saver.to_json())
			
		if payload['contribute'] == 'agree':
				
			# serialize weights to HDF5
			weights = os.path.join(RESOURCES, payload['weights_file'])
			model_saver.save_weights(weights)
			print("Saved model to disk")
		'''
		if payload['contribute'] == 'agree':
				
			# save into file
			saved_model = os.path.join(RESOURCES, payload['model_name'] + '.pkl')
			pickle.dump(model, open(saved_model, 'wb')) 

			print("Saved model to disk")

		if (payload['mode']=='classification'):
			result = getClassificationScore(payload['model_name'], payload['metrics'], y_test, y_pred, y_prob)
		
		elif (payload['mode']=='regression'):
			result = getRegressionScore(payload['model_name'], payload['metrics'], y_test, y_pred)

		#results.append(result)
		results[payload['metrics'].upper()].append(result)


	return results

def runML_old(payload):

	path_to_file = os.path.join(RESOURCES, payload['filename'])
	df = readFile(path_to_file)

	df.drop(payload['drops'], axis=1, inplace=True)		#For now it is still just dropping
	df = impute(df, payload['missing'])

	X, y = makeDataset(df, payload['targets'])
	X = scaling(X, payload['normalization'])

	models = []
	df = {}
	names = []
	accuracies = []
	precisions = []
	recalls = []
	f1s = []
	rocs = []
	mccs = []

	results = []

	#confusions = []

	y_preds = []
	y_probs = []
	roc_curves = []

	if payload['mode'] == 'classification':
		'''
		models.append(('SVM', SVC(gamma='auto', probability=True)))
		#models.append(('Linear SVM', LinearSVC()))
		models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='auto')))
		models.append(('kNN', KNeighborsClassifier()))
		models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
		models.append(('Naive Bayes', GaussianNB()))
		#models.append(('Multi-layer Perceptron', MLPClassifier()))
		'''
		if payload['validation'] == 'holdout':

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['holdout']))

			for name, model in models:
				#model.fit(X_train, y_train)
				model.fit(X_train, y_train.values.ravel())

				y_pred = model.predict(X_test)
				y_prob = model.predict_proba(X_test)[:, 1]

				y_preds.append(y_pred)
				y_probs.append(y_prob)

				fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=2)
				roc_curves.append((fpr, tpr))

				names.append(name)
				accuracies.append(accuracy_score(y_test, y_pred))
				precisions.append(precision_score(y_test, y_pred))
				recalls.append(recall_score(y_test, y_pred))
				f1s.append(f1_score(y_test, y_pred))
				rocs.append(roc_auc_score(y_test, y_prob))
				mccs.append(matthews_corrcoef(y_test, y_pred))

				results.append(getScore(name, payload['metrics'], y_test, y_pred, y_prob))

				#confusions.append(confusion_matrix(y_test, y_pred))
		else:

			scoring = payload['metrics']

			for name, model in models:
				crossval = cross_validate(estimator=model, X=X, y=y, cv=1/float(payload['test-size']), scoring=scoring)
				print(crossval)

		df['model_name'] = names
		df['accuracy'] = accuracies
		df['precision'] = precisions
		df['recall'] = recalls
		df['f1'] = f1s
		df['rocs'] = rocs
		df['mccs'] = mccs

		dataframe = pd.DataFrame(df)
		dataframe.sort_values('accuracy', ascending=False, inplace=True)

		session['test'] = y_test.values.ravel().tolist()
		#session['preds'] = y_preds.values.tolist()
		#session['probs'] = y_probs.values.tolist()

	return results

'''
def runNN(payload, compare, tuning_params, index=0, scaled_first=True, split_first=True):

	#results = []
	print(payload['metrics'])

	name = payload['model-name']
	model = None
	print(tuning_params)

	if payload['hyper-param'] and payload['mode']=='Classification':
		model = KerasClassifier(build_fn=createClassifier, loss_func='binary_crossentropy', opt_func='adam', act_hidden='relu', act_output='sigmoid')
	elif payload['hyper-param'] and payload['mode']=='Regression':
		model = KerasRegressor(build_fn=createRegressor, loss_func='mean_squared_error', opt_func='adam', act_hidden='relu', act_output='linear')
	elif not payload['hyper-param'] and payload['mode']=='Classification':
		model = KerasClassifier(build_fn=createClassifier,
					loss_func='binary_crossentropy', opt_func='adam', 
					batch_size=tuning_params['batch_size'],
					epochs=tuning_params['epochs'],
					num_hidden=tuning_params['num_hidden'],
					node_hidden=tuning_params['node_hidden'],
					act_hidden='relu', act_output='sigmoid')
	elif not payload['hyper-param'] and payload['mode']=='Regression':
		model = KerasRegressor(build_fn=createRegressor, 
                    loss_func='mean_squared_error', opt_func='adam', 
                    batch_size=tuning_params['batch_size'],
                    epochs=tuning_params['epochs'],
                    num_hidden=tuning_params['num_hidden'],
                    node_hidden=tuning_params['node_hidden'],
                    act_hidden='relu', act_output='linear')


	#name = TUNABLE_MODELS[0][0] if payload['hyper-param'] else NO_TUNABLE_MODELS[0][0]
	#model = TUNABLE_MODELS[0][1] if payload['hyper-param'] else NO_TUNABLE_MODELS[0][1]
	print('Running', name, model, ', tuning hyperparameter:', payload['hyper-param'])

	complete_filename = os.path.join(RESOURCES, payload['filename'])
	df = readFile(complete_filename)

	if payload['mode']=='Regression':
		df = df[df[payload['filter']]==payload['selected_condition']]

	df.drop(payload['drops'], axis=1, inplace=True)		#drop first, so NaN could be minimized
	if payload['missing'] == 'drop':
		df.dropna(inplace=True)

	params = generateParams(payload)

	X, y = makeDataset(df, payload['targets'])
	
	#For now, multi-label classification is not supported
	if (len(payload['targets'])>1):
		return ('For now, multi-label classification is not supported! Exiting...')

	y = y.ravel()
	metrics = payload['metrics']
	if payload['mode'] == 'Classification':
		y = y - 1

	#if scaled_first:
	#	X = StandardScaler().fit_transform(X)
	if split_first:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['test-size']))
	else:
		X_train, X_test, y_train, y_test = X, X, y, y

	#params_copy = params.copy()
	params.append(('mod', model))
	pipeline = Pipeline(params)

	if payload['hyper-param'] is not None:

		if payload['tuning'] == 'grids':
			exe = GridSearchCV(pipeline, tuning_params, cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)
		elif payload['tuning'] == 'randoms':
			exe = RandomizedSearchCV(pipeline, tuning_params, cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)
		else:
			return ('This part for Bayesian Optimization, or Swarm Intelligence... Exiting!')

	else:

		if payload['crossval'] is not None:

			scoring = metrics.copy()
			#print(scoring)
			if payload['mode']=='Regression':
				for n, i in enumerate(scoring):
					if i == 'rmse':
						scoring[n] = 'neg_mean_squared_error'
					if i == 'mae':
						scoring[n] = 'neg_mean_absolute_error'

			#print(scoring)

			start = datetime.now()
			res = cross_validate(estimator=pipeline, X=X_train, y=y_train, cv=int(1/float(payload['test-size'])), scoring=scoring)
			end = datetime.now()
			print('Total execution time:', str(end-start))

			res_dict = {}
			res_dict['Model Name'] = name
			for s in scoring:
				key = 'test_' + s
				res_dict[key] = res[key].mean()

			return res_dict
			#res['test_precision'].mean(), res['test_recall'].mean(), res['test_f1'].mean(), res['test_roc_auc'].mean()

		else:

			exe = pipeline

	start = datetime.now()
	exe.fit(X_train, y_train)
	end = datetime.now()
	print('Total execution time:', str(end-start))

	if payload['hyper-param']:

		top3 = pd.DataFrame(exe.cv_results_)
		top3.sort_values(by='rank_test_score', inplace=True)
		print(top3)

		best_params = exe.best_params_
		print('Best config:', best_params)
		y_pred = exe.best_estimator_.predict(X_test)
		
		if payload['save-best-config']:
			json_str = json.dumps(exe.best_params_)
			best_config_json = os.path.join(RESOURCES, payload['best-config-file'])
			with open(best_config_json, 'w') as json_file:
				json_file.write(json_str)

		if payload['mode']=='Classification':
			y_prob = exe.best_estimator_.predict_proba(X_test)[:, 1]
			
		model_saver = exe.best_estimator_['mod'].model

	else:
		y_pred = exe.predict(X_test)
		if payload['mode']=='Classification':
			y_prob = exe.predict_proba(X_test)[:, 1]
			
		model_saver = pipeline.named_steps['mod'].model
	
	if (payload['save-architecture']):
	
		architecture = os.path.join(RESOURCES, payload['architecture-file'])
		
		with open(architecture, 'w') as json_file:
			json_file.write(model_saver.to_json())
		
	if (payload['save-weights']):
    		
		# serialize weights to HDF5

		weights = os.path.join(RESOURCES, payload['weights-file'])
		model_saver.save_weights(weights)
		print("Saved model to disk")
	

	if (payload['mode']=='Classification'):
		results = getClassificationScore(name, metrics, y_test, y_pred, y_prob)
	
	elif (payload['mode']=='Regression'):
		results = getRegressionScore(name, metrics, y_test, y_pred)

	return results
'''