import os
import shutil
import pandas as pd

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPDIR = os.path.join(APP_ROOT, '../resources/temp/')
DEPLOYED = os.path.join(APP_ROOT, '../resources/deployed/')
DB = os.path.join(APP_ROOT, '../resources/db/')

db_name = 'garuda-model-persist-20190922.csv'

def moveModel(models):

    for model in models:

        to_move = os.path.join(TEMPDIR, model + '.pkl')
        shutil.move(to_move, DEPLOYED)

def clearTemp():

	for pkl in os.listdir(TEMPDIR):

		to_remove = os.path.join(TEMPDIR, pkl)
		os.remove(to_remove)

def writeDB(payload):

	db = os.path.join(DB, db_name)

	data = [i.split(',') for i in payload['vals'].split(';')]
	columns = payload['cols'].split(',')

	df = pd.DataFrame(data=data, columns=columns)
	df['Created At'] = payload['timestamp']
	df['Metrics'] = columns[3]
	df['Cross-validated'] = payload['cv']
	df['Hyperparameter Tuned'] = payload['hy']
	df['Owner'] = payload['owner']
	df['Description'] = payload['description']
	df.rename(columns={columns[3]: 'Performance'}, inplace=True)

	print(df.columns)

	df = df[['Model Name', 'Created At', 'Mode', 'Algorithm', 'Metrics',
							'Performance', 'Cross-validated', 'Hyperparameter Tuned',
							'Owner', 'Description']]
	df = df[df['Model Name'].isin(payload['deploy'])]

	#df = pd.DataFrame(columns=['Model name', 'Created at', 'Mode', 'Algorithm', 'Metrics',
	#							'Performance', 'Cross-validated', 'Hyper-paramater tuned',
	#							'Owner', 'Description'])


	if os.path.exists(db):
		dfd = pd.read_csv(db)
		df = dfd.append(df)

	df.to_csv(db, index=False)
