import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')

def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def getDataType(filename):

	complete_filename = os.path.join(RESOURCES, filename)
	df = readFile(complete_filename)

	val = []
	flag = []
	for col in df.columns:
		if df[col].dtype == np.float64 or df[col].dtype == np.int64:
			val.append('numeric')
			flag.append(0)			#0 for numeric
		#elif df[col].dtype == np.int64:
		#	val.append('categorical')
		else:
			try:
				df[col] = pd.to_datetime(df[col])
				val.append('datetime')
				flag.append(1)		#1 for datetime
			except:
				val.append('text')
				flag.append(2)		#2 for text

	df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val, 'flag': flag})
	df2 = df.describe().round(2).transpose().reset_index().rename(columns={'index': 'column'})
	unique = df.nunique().reset_index().rename(columns={'index': 'column', 0: 'unique'})
	nan = df.isnull().sum().reset_index().rename(columns={'index': 'column', 0: 'missing'})
	
	df3 = pd.merge(df1, unique, how='left')
	df3 = pd.merge(df3, nan, how='left')
	df3 = pd.merge(df3, df2, how='left')
	
	df3.fillna('', inplace=True)

	return df3

def readFile(filename):

	#complete_filename = os.path.join(RESOURCES, filename)
	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

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
		params.append(('encoder', encoder))
	if payload['normalization'] != 'none':
		if payload['normalization'] == 'minmax':
			scaler = MinMaxScaler()
		elif payload['normalization'] == 'standard':
			scaler = StandardScaler()
		elif payload['normalization'] == 'robust':
			scaler = RobustScaler()
		params.append(('scaler', scaler))
	if payload['dim_red'] != None and payload['num_of_dim'] != None:
		params.append(('reducer', PCA(n_components=int(payload['num_of_dim']))))

	return params

def makeDataset(df, targets):

	X = df.drop(targets, axis=1)
	y = df[targets]

	return X, y

def getScore(name, scores, test, pred, prob):

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

	return name, acc, prec, rec, f1, roc

def createModel(loss_func, opt_func, num_hidden, node_hidden, act_hidden, act_output):

	model = Sequential()

	for i in range(num_hidden):
		model.add(Dense(node_hidden, activation=act_hidden))
	
	model.add(Dense(2, activation=act_output))
	model.compile(loss=loss_func, optimizer=opt_func, metrics=['accuracy'])
	
	return model

def runNN(payload, compare):

	CLASSIFIERS = []
	HYPERPARAMS = []

	if payload['hyper-param'] is None:
		CLASSIFIERS = [
			('Neural Networks', KerasClassifier(build_fn=createModel, epochs=50, batch_size=32,
												loss_func='categorical_crossentropy', opt_func='adam',
												num_hidden=1, node_hidden=16, act_hidden='relu',
												act_output='softmax')),

			('Random Forest', RandomForestClassifier(n_estimators=100))
		]
	else:
		batch_sizes = [32]
		epochs = [50]
		node_hidden = [16, 32]
		num_hidden = [2, 4]
		nn_grid = {'clf__batch_size': batch_sizes,
				'clf__epochs': epochs,
				'clf__node_hidden': node_hidden,
				'clf__num_hidden': num_hidden}
		
		n_estimators = [100, 500, 1000]
		max_depth = [10, 50, 100]
		rf_grid = {'n_estimators': n_estimators,
				'max_depth': max_depth}

		CLASSIFIERS = [
			('Neural Networks', KerasClassifier(build_fn=createModel,
												loss_func='categorical_crossentropy', opt_func='adam',
												act_hidden='relu', act_output='softmax')),
			('Random Forest', RandomForestClassifier())
		]
		HYPERPARAMS = [
			('Neural Networks', nn_grid),
			('Random Forest', rf_grid)
		]

	complete_filename = os.path.join(RESOURCES, payload['filename'])
	df = readFile(complete_filename)

	df.drop(payload['drops'], axis=1, inplace=True)
	if payload['missing'] == 'drop':
		df.dropna(inplace=True)

	params = generateParams(payload)

	X, y = makeDataset(df, payload['targets'])

	#df = {}
	#names = []
	accuracies = []
	precisions = []
	recalls = []
	f1s = []
	rocs = []
	#mccs = []

	results = []

	#y_preds = []
	#y_probs = []
	#roc_curves = []

	crossvals = []

	#if payload['crossval'] is None:

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['test-size']))

	for i in range(1):

		name = CLASSIFIERS[i][0]
		classifier = CLASSIFIERS[i][1]

		params_copy = params.copy()
		params_copy.append(('clf', classifier))
		pipeline = Pipeline(params_copy)

		if payload['hyper-param'] is None and payload['crossval'] is not None:

			scoring = payload['metrics']

			crossval = cross_validate(estimator=pipeline, X=X_train.values, y=y_train.values.ravel(), cv=int(1/float(payload['test-size'])), scoring=scoring)
			
			#crossval2 = cross_val_score(estimator=pipeline, X=X.values, y=y.values.ravel(), cv=int(1/float(payload['test-size'])))

			accuracies.append(crossval['test_accuracy'])

			#accuracies.append(crossval2)

			#precisions.append(crossval['test_precision'].mean())
			#recalls.append(crossval['test_recall'].mean())
			#f1s.append(crossval['test_f1'].mean())
			#rocs.append(crossval['test_roc_auc'].mean())

		else:

			if payload['hyper-param'] is not None:

				if payload['tuning'] == 'grids':
					print('Tuning with Grid!')
					exe = GridSearchCV(pipeline, HYPERPARAMS[i][1], cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)
				else:
					print('Tuning with Random!')
					exe = RandomizedSearchCV(pipeline, HYPERPARAMS[i][1], cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)

				exe.fit(X.values, y.values.ravel())

				best_score = exe.best_score_
				best_params = exe.best_params_

				print(best_score)
				print(best_params)
				print(list(best_params.keys()))

				y_pred = exe.best_estimator_.predict(X_test)
				y_prob = exe.best_estimator_.predict_proba(X_test)[:, 1]

				cv_results = exe.cv_results_

				print('------------')
				print(exe.best_score_, exe.best_params_)
				means = exe.cv_results_['mean_test_score']
				parameters = exe.cv_results_['params']
				print('------------')
				for mean, parameter in zip(means, parameters):
					print(mean, parameter)
				
				print('------------')

				rank = list(exe.cv_results_['rank_test_score'])
				top3idx = [rank.index(1), rank.index(2), rank.index(3)]
				params = exe.cv_results_['params']
				topParams = [params[rank.index(1)],
								params[rank.index(2)],
								params[rank.index(3)]
							]

			elif payload['crossval'] is None:

				exe = pipeline
			'''
			exe.fit(X_train.values, y_train.values.ravel())

			#if payload['hyper-param'] is not None: 
			#	print(exe.best_score_)    
			#	print(exe.best_params_)

			y_pred = exe.predict(X_test)
			y_prob = exe.predict_proba(X_test)[:, 1]

			#y_preds.append(y_pred)
			#y_probs.append(y_prob)

			#fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=2)
			#roc_curves.append((fpr, tpr))

			#names.append(name)

			accuracies.append(accuracy_score(y_test, y_pred))
			#precisions.append(precision_score(y_test, y_pred))
			#recalls.append(recall_score(y_test, y_pred))
			#f1s.append(f1_score(y_test, y_pred))
			#rocs.append(roc_auc_score(y_test, y_prob))
			'''

			results.append(getScore(name, payload['metrics'], y_test, y_pred, y_prob))

	#df['model_name'] = names
	#df['accuracy'] = accuracies
	#df['precision'] = precisions
	#df['recall'] = recalls
	#df['f1'] = f1s
	#df['rocs'] = rocs
	#df['mccs'] = mccs

	#print(accuracies)

	columns = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

	#dataframe = pd.DataFrame(df)
	#dataframe.sort_values('accuracy', ascending=False, inplace=True)

	return columns, results, cv_results, top_3