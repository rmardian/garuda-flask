from flask import session
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta

#from sklearn.externals import joblib 
from sklearn.impute import SimpleImputer
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

from app.mod_ml.model import CLASSIFIERS, REGRESSORS, PARAMS

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')
TEMPDIR = os.path.join(APP_ROOT, '../resources/temp/')
DEPLOYED = os.path.join(APP_ROOT, '../resources/deployed/')

def readFile(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

def makeDataset(df, target):

	X = df.drop(target, axis=1)
	y = df[target]

	return X.values, y.values.ravel()

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

def getClassificationScore(name, score, test, pred, prob):
	'''
	acc, prec, rec, f1, roc = None, None, None, None, None
	
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
	'''

	if score == 'accuracy':
		return accuracy_score(test, pred)
	elif score == 'precision':
		return precision_score(test, pred)
	elif score == 'recall':
		return recall_score(test, pred)
	elif score == 'f1':
		return f1_score(test, pred)
	elif score == 'roc_auc':
		return roc_auc_score(test, prob)

def getRegressionScore(name, score, pred, test):
	'''
	mae, mse, r2 = None, None, None

	for score in scores:
		if score == 'mae':
			mae = mean_absolute_error(test, pred)
		elif score == 'mse':
			mse = mean_squared_error(test, pred)
		elif score == 'rmse':
			rmse = np.sqrt(mean_squared_error(test, pred))
		elif score == 'r2':
			r2 = r2_score(test, pred)

	score_dict = {
		'Mode': 'Regression',
		'Model Name': name,
		'Mean Absolute Error': mae,
		'Mean Squared Error': mse,
		'R-squared': r2
	} 
	return {k:[v] for k,v in score_dict.items() if v is not None}
	'''

	if score == 'neg_mean_absolute_error':
		return mean_absolute_error(test, pred)
	elif score == 'neg_mean_squared_error':
		return mean_squared_error(test, pred)
	elif score == 'r2':
		return r2_score(test, pred)

def runML(payload):

	cv = 'No'
	hy = 'No'

	path_to_file = os.path.join(RESOURCES, payload['filename'])
	df = readFile(path_to_file)

	df.drop(payload['drops'], axis=1, inplace=True)		#drop first, so NaN could be minimized
	if payload['missing'] == 'drop':
		df.dropna(inplace=True)
	X, y = makeDataset(df, payload['target'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['holdout']))

	results = {}
	results['Model Name'] = []
	results['Mode'] = []
	results['Algorithm'] = []
	results[payload['metrics'].upper()] = []

	MODELS = None
	if (payload['mode']=='classification'):
		MODELS = CLASSIFIERS.copy()
	elif (payload['mode']=='regression'):
		MODELS = REGRESSORS.copy()

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
			cv = 'Yes'

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

		start = datetime.now()
		exe.fit(X_train, y_train)
		end = datetime.now()
		print('Total execution time:', str(end-start))

		if payload['tuning'] != 'none':

			hy = 'Yes'
			cv = 'Yes'

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
				
			model_saver = exe.best_estimator_	#in keras model = exe.best_estimator_['model'].model

		else:
			y_pred = exe.predict(X_test)
			if payload['mode']=='classification':
				y_prob = exe.predict_proba(X_test)[:, 1]

			model_saver = exe.named_steps[idx]
			print(model_saver)

		pkl_name = payload['model_name'] + '-' + str(i) + '.pkl'
		
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
			saved_model = os.path.join(TEMPDIR, pkl_name)
			pickle.dump(exe, open(saved_model, 'wb')) 

			print("Saved model to disk")

		if (payload['mode']=='classification'):
			result = getClassificationScore(payload['model_name'], payload['metrics'], y_test, y_pred, y_prob)
		
		elif (payload['mode']=='regression'):
			result = getRegressionScore(payload['model_name'], payload['metrics'], y_test, y_pred)

		results[payload['metrics'].upper()].append(result)

	return results, cv, hy