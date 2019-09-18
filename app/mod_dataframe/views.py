from flask import Blueprint, render_template

dataframe_blueprint = Blueprint('dataframe', __name__)

from app.mod_dataframe.controllers import read_mongo
from config import CONFIG

@dataframe_blueprint.route('/df')
def df_index():
    return render_template('dataframe/index.html')

@dataframe_blueprint.route('/dfload')
def df_load():

    df = read_mongo(CONFIG['database'], 'film')

    print(df.shape)
    print(df.head(10))

    return render_template('dataframe/success.html')


