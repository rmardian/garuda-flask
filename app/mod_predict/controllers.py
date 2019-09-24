import pandas as pd
import os
import pickle


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')
DEPLOYED = os.path.join(APP_ROOT, '../resources/deployed/')


def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def readFile(filename):

	df = pd.read_csv(filename)
	#df.rename(columns={'index': 'date'}, inplace=True)

	return df

def runPrediction(filename, modelName):

	path_to_file = os.path.join(RESOURCES, filename)

	df = readFile(path_to_file)

	path_to_model = os.path.join(DEPLOYED, modelName + '.pkl')
	loaded_model = pickle.load(open(path_to_model, 'rb'))

	pred = loaded_model.predict(df.values)

	return pred