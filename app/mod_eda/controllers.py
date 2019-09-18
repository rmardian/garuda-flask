from flask import session
import pandas as pd
import numpy as np
import os
import json

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

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

def impute(df, missing):

	if missing == 'drop':
		df.dropna(inplace=True)

	else:
		imputer = Imputer(missing_values='NaN', strategy=missing, axis=0)
		imputer = imputer.fit(df)
		df = imputer.transform(df)

	return df

def featureDefinition(df, targets):

	X = df.drop(targets, axis=1)
	y = df[targets]

	return X, y

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

def runML(payload):

	complete_filename = os.path.join(RESOURCES, payload['filename'])
	df = readFile(complete_filename)
	df = impute(df, payload['missing'])
	df.drop(payload['drops'], axis=1, inplace=True)		#For now it is still just dropping
	X, y = featureDefinition(df, payload['targets'])
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

	if payload['mode'] == 'supervised':

		models.append(('SVM', SVC(gamma='auto', probability=True)))
		#models.append(('Linear SVM', LinearSVC()))
		models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='auto')))
		models.append(('kNN', KNeighborsClassifier()))
		models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
		models.append(('Naive Bayes', GaussianNB()))
		models.append(('Multi-layer Perceptron', MLPClassifier()))

		if payload['crossval'] is None:

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['test-size']))
			print(len(y_test) / (len(y_train) + len(y_test)))

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

	return dataframe, y_test, y_preds, y_probs, results, roc_curves




	


