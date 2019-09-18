from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "1234567890"

from config import CONFIG, SOURCE
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_auth.views import auth_blueprint
from app.mod_load.views import load_blueprint
from app.mod_clotho.views import clotho_blueprint
from app.mod_parser.views import parser_blueprint
from app.mod_query.views import query_blueprint
from app.mod_eda.views import eda_blueprint
from app.mod_ml.views import ml_blueprint
from app.mod_NN.views import nn_blueprint

from app.mod_dataframe.views import dataframe_blueprint
from app.mod_denormalization.views import denormalization_blueprint
from app.mod_correlation.views import correlation_blueprint

from app.mod_auth.controllers import log_in, create_user
from app.mod_load.controllers import generate, load
from app.mod_clotho.controllers import createPart, createDevice, deleteDevice
from app.mod_query.controllers import resolveSearch
from app.mod_parser.controllers import allowedFile, resolveImport

app.register_blueprint(auth_blueprint, url_prefix='/auth')
app.register_blueprint(load_blueprint, url_prefix='/load')
app.register_blueprint(clotho_blueprint, url_prefix='/clotho')
app.register_blueprint(parser_blueprint, url_prefix='/parser')
app.register_blueprint(query_blueprint, url_prefix='/query')
app.register_blueprint(eda_blueprint, url_prefix='/eda')
app.register_blueprint(ml_blueprint, url_prefix='/ml')
app.register_blueprint(nn_blueprint, url_prefix='/neural-net')

app.register_blueprint(dataframe_blueprint, url_prefix='/df')
app.register_blueprint(denormalization_blueprint, url_prefix='/denormalized')

#@app.route("/ingest")
#def ingest():
#  return 'Ingest home!'

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
	return redirect(url_for('recommender'))

@app.route("/login.html", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
def login():

	if request.method == 'POST':
		session_user, authHeader = log_in(request.form['username'], request.form['password'])
		if session_user != None:
			session['user'] = session_user
			session['logged_in'] = True
			session['authHeader'] = authHeader
			return redirect(url_for('search'))
		return render_template('login.html', message="Login error! Please check your username and password")
	return render_template('login.html')

@app.route("/register", methods=['POST'])
def register():

	if request.method == 'POST':
		session_user, authHeader = create_user(request.form['username'], request.form['password'], request.form['email'], request.form['name'])
		if session_user != None:
			session['user'] = session_user
			session['logged_in'] = True
			session['authHeader'] = authHeader
			return redirect(url_for('search'))
		return render_template('login.html', message="Error: Username or email has been taken!")
	return render_template('login.html')

@app.route('/logout.html', methods=['GET', 'POST'])
@app.route('/logout', methods=['GET', 'POST'])
def logout():
	session.pop('user', None)
	session.pop('logged_in', None)
	session.pop('authHeader', None)
	return redirect(url_for('login'))

@app.route('/search.html', methods=['GET', 'POST'])
@app.route('/search', methods=['GET', 'POST'])
def search():

	if request.method == 'POST':

		payloads = {}
		if request.form['biodesignId'] != None and request.form['biodesignId'] != "":
			payloads['biodesignId'] = request.form['biodesignId']
		if request.form['displayId'] != None and request.form['displayId'] != "":
			payloads['displayId'] = request.form['displayId']
		if request.form['name'] != None and request.form['name'] != "":
			payloads['name'] = request.form['name']
		if request.form['role'] != None and request.form['role'] != "":
			payloads['role'] = request.form['role']
		if request.form['type'] != None and request.form['type'] != "":
			payloads['type'] = request.form['type']
		if request.form['sequence'] != None and request.form['sequence'] != "":
			payloads['sequence'] = request.form['sequence']
		if request.form['userSpace'] != None and request.form['userSpace'] != "":
			payloads['userSpace'] = request.form['userSpace']

		message, results = resolveSearch(payloads, session['user'], session['authHeader'])
		#print(json.dumps(results, indent=4))

		return render_template('result.html', message=message, results=results)

	# IF REQUEST METHOD = GET
	if 'user' in session and session['logged_in'] == True:
		return render_template('search.html')

	# IF NOT LOGIN
	return redirect(url_for('login'))

@app.route('/result.html', methods=['GET', 'POST'])
@app.route('/result', methods=['GET', 'POST'])
def result():
	if 'user' in session and session['logged_in'] == True:
		return redirect(url_for('search'))
	return redirect(url_for('login'))

@app.route('/add.html', methods=['GET', 'POST'])
@app.route('/add', methods=['GET', 'POST'])
def add():

	if request.method == 'POST':

		payloads = {}
		if request.form['name'] != None and request.form['name'] != "":
			payloads['name'] = request.form['name']
		if request.form['displayId'] != None and request.form['displayId'] != "":
			payloads['displayId'] = request.form['displayId']
		if request.form['role'] != None and request.form['role'] != "":
			payloads['role'] = request.form['role']
		if request.form['sequence'] != None and request.form['sequence'] != "":
			payloads['sequence'] = request.form['sequence']
		if request.form['parameters'] != None and request.form['parameters'] != "":
			payloads['parameters'] = "[" + request.form['parameters'] + "]"
		elif request.form['parameters'] == "":
			payloads['parameters'] = "[]"

		results = createPart(payloads, session['user'], session['authHeader'])
		#print(json.dumps(results, indent=4))

		return render_template('add.html', message="Created a part (ID: " + results + ")")

	# IF REQUEST METHOD = GET
	if 'user' in session and session['logged_in'] == True:
		return render_template('add.html')

	# IF NOT LOGIN
	return redirect(url_for('login'))

@app.route('/add_device.html', methods=['GET', 'POST'])
@app.route('/add_device', methods=['GET', 'POST'])
def add_device():

	if request.method == 'POST':

		payloads = {}
		if request.form['name'] != None and request.form['name'] != "":
			payloads['name'] = request.form['name']
		if request.form['displayId'] != None and request.form['displayId'] != "":
			payloads['displayId'] = request.form['displayId']
		if request.form['role'] != None and request.form['role'] != "":
			payloads['role'] = request.form['role']
		if request.form['createSeqFromParts'] != None and request.form['createSeqFromParts'] != "":
			payloads['createSeqFromParts'] = request.form['createSeqFromParts']
		if request.form['parameters'] != None and request.form['parameters'] != "":
			payloads['parameters'] = "[" + request.form['parameters'] + "]"
		elif request.form['parameters'] == "":
			payloads['parameters'] = "[]"
		if request.form['partIds'] != None and request.form['partIds'] != "":
			payloads['partIds'] = "[" + request.form['partIds'] + "]"
		elif request.form['partIds'] == "":
			payloads['partIds'] = "[]"

		results = createDevice(payloads, session['user'], session['authHeader'])
		#print(json.dumps(results, indent=4))

		return render_template('add_device.html', message="Created a device (ID: " + results + ")")

	# IF REQUEST METHOD = GET
	if 'user' in session and session['logged_in'] == True:
		return render_template('add_device.html')

	# IF NOT LOGIN
	return redirect(url_for('login'))

@app.route('/import.html', methods=['GET', 'POST'])
@app.route('/import', methods=['GET', 'POST'])
def importer():

	if request.method == 'POST':

		if 'file' not in request.files:
			print(request.url)
			return render_template('import.html', message="Error: No file is found!")

		file = request.files['file']
		if file.filename == "":
			return render_template('import.html', message="Error: No file is selected!")

		if file and allowedFile(file.filename):

			target = os.path.join(APP_ROOT, 'resources/inputs/')

			#if not os.path.isdir(target):
			#	os.mkdir(target)
			#for multiple files
			#for file in request.files.getlist("file"):
			#	print(file.filename)

			filename = secure_filename(file.filename)

			#destination = "/".join([target, filename])
			#file.save(destination)
			file.save(os.path.join(target, filename))

			results = resolveImport(os.path.join(target, filename), session['user'], session['authHeader'])

			return render_template('import.html', message=results)

	# IF REQUEST METHOD = GET
	if 'user' in session and session['logged_in'] == True:
		return render_template('import.html')

	# IF NOT LOGIN
	return redirect(url_for('login'))

@app.route('/recommender.html', methods=['GET', 'POST'])
@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
	if 'user' in session and session['logged_in'] == True:
		return render_template('recommender.html')
	return redirect(url_for('login'))

@app.route('/delete', methods=['POST'])
def delete():

	#deleteDevice(request.json['id'], session['user'], session['authHeader'])
	return "SUCCESS!!"

@app.errorhandler(404) 
def not_found(e): 
  return render_template("404.html") 

@app.route('/')
def main():
  return render_template('index.html')

@app.errorhandler(405)
def method_not_allowed(e):
  return render_template("404.html") 

@app.route('/goto/<api>', methods=['GET', 'POST'])
def goto(api):

  if request.method == "POST":

    return redirect(url_for(api + '.html'))

@app.route('/pipeline', methods=['GET', 'POST'])
def pipeline():

  if request.method == "POST":

    print('***POST PIPELINE***')
    '''
    tables = get_tables(SOURCE)
    for i in range(len(tables)):
        columns = get_schema(SOURCE, tables[i])
        data = get_data(SOURCE, tables[i])
        payloads = generate(columns, data)
        load(CONFIG, tables[i], payloads)
    '''

    return render_template('success.html')

  return render_template('pipeline.html')
