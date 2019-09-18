import pandas as pd
import numpy as np

def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def getDataType(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	val = []
	for col in df.columns:
		if df[col].dtype == np.float64:
			val.append('numeric')
		elif df[col].dtype == np.int64:
			val.append('categorical')
		else:
			try:
				df[col] = pd.to_datetime(df[col])
				val.append('datetime')
			except:
				val.append('text')

	df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val})
	df2 = df.describe().round(2).transpose().reset_index().rename(columns={'index': 'column'})
	unique = df.nunique().reset_index().rename(columns={'index': 'column', 0: 'unique'})
	nan = df.isnull().sum().reset_index().rename(columns={'index': 'column', 0: 'missing'})
	
	df3 = pd.merge(df1, unique, how='left')
	df3 = pd.merge(df3, nan, how='left')
	df3 = pd.merge(df3, df2, how='left')
	
	df3.fillna('', inplace=True)

	return(df3)