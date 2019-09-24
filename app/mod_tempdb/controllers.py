import pandas as pd
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(APP_ROOT, '../resources/db/')
db_name = 'garuda-model-persist-20190922.csv'

def populateDeployedModel():

	options = []
	db_path = os.path.join(DB, db_name)
	
	if os.path.exists(db_path):
		df = pd.read_csv(db_path)
		models = df['Model Name'].tolist()
		algos = df['Algorithm'].tolist()

		for i in range(len(models)):
			options.append((models[i], algos[i]))

	return options

def generateModelTable():

	table = []
	db_path = os.path.join(DB, db_name)
	
	if os.path.exists(db_path):

		df = pd.read_csv(db_path)
		models = df['Model Name'].tolist()

		for model in models:
			df_temp = df[df['Model Name']==model].transpose()
			table.append(df_temp.reset_index().values)

	return table


	


