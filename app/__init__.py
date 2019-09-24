from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "06251987"

from app.mod_auth.views import auth_blueprint
from app.mod_clotho.views import clotho_blueprint
from app.mod_parser.views import parser_blueprint
from app.mod_eda.views import eda_blueprint
from app.mod_ml.views import ml_blueprint
from app.mod_deploy.views import deploy_blueprint
from app.mod_predict.views import predict_blueprint

from app.mod_tempdb.controllers import populateDeployedModel, generateModelTable

app.register_blueprint(auth_blueprint, url_prefix='/auth')
app.register_blueprint(clotho_blueprint, url_prefix='/clotho')
app.register_blueprint(parser_blueprint, url_prefix='/parser')
app.register_blueprint(eda_blueprint, url_prefix='/eda')
app.register_blueprint(ml_blueprint, url_prefix='/ml')
app.register_blueprint(deploy_blueprint, url_prefix='/deploy')
app.register_blueprint(predict_blueprint, url_prefix='/predict')

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def index():
	'''
	if 'user' in session and session['logged_in'] == True:
		return redirect(url_for('recommender'))
	return redirect(url_for('login'))
	'''
	session['user'] = 'garuda'
	session['logged_in'] = True
	return redirect(url_for('train'))

@app.route('/train.html', methods=['GET', 'POST'])
@app.route('/train', methods=['GET', 'POST'])
def train():
	if 'user' in session and session['logged_in'] == True:
		return render_template('train.html')
	return redirect(url_for('auth.login'))

@app.route('/test.html', methods=['GET', 'POST'])
@app.route('/test', methods=['GET', 'POST'])
def test():
	if 'user' in session and session['logged_in'] == True:
		return render_template('test.html', models=populateDeployedModel(),
								tables=generateModelTable())
	return redirect(url_for('auth.login'))

@app.route('/clustering.html', methods=['GET', 'POST'])
@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
	if 'user' in session and session['logged_in'] == True:
		return render_template('login.html')
	return redirect(url_for('auth.login'))

@app.errorhandler(404) 
def not_found(e): 
	return render_template("page_404.html") 

#this still does not work correctly, the idea is to make it possible to generate and utilize API on demand
#is there any use of this?
@app.route('/goto/<api>', methods=['GET', 'POST'])
def goto(api):

	return redirect(url_for(api))

#not sure what this pipeline does, commenting it for now
'''
@app.route('/pipeline', methods=['GET', 'POST'])
def pipeline():

	if request.method == "POST":
		print('***POST PIPELINE***')
		tables = get_tables(SOURCE)
		for i in range(len(tables)):
			columns = get_schema(SOURCE, tables[i])
			data = get_data(SOURCE, tables[i])
			payloads = generate(columns, data)
			load(CONFIG, tables[i], payloads)

		return render_template('success.html')

	return render_template('pipeline.html')

'''