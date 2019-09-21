from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os

clotho_blueprint = Blueprint('clotho', __name__)

from app.mod_clotho.controllers import createPart, createDevice, deleteDevice
from app.mod_query.controllers import resolveSearch
from app.mod_parser.controllers import allowedFile, resolveImport

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@clotho_blueprint.route('/clotho')
def clotho():
	return 'Clotho!'

@clotho_blueprint.route('/search.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/search', methods=['GET', 'POST'])
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

@clotho_blueprint.route('/add.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/add', methods=['GET', 'POST'])
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

@clotho_blueprint.route('/add_device.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/add_device', methods=['GET', 'POST'])
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

@clotho_blueprint.route('/import.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/import', methods=['GET', 'POST'])
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

@clotho_blueprint.route('/result.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/result', methods=['GET', 'POST'])
def result():
	if 'user' in session and session['logged_in'] == True:
		return redirect(url_for('search'))
	return redirect(url_for('login'))

@clotho_blueprint.route('/recommender.html', methods=['GET', 'POST'])
@clotho_blueprint.route('/recommender', methods=['GET', 'POST'])
def recommender():
	if 'user' in session and session['logged_in'] == True:
		return render_template('recommender.html')
	return redirect(url_for('login'))

@clotho_blueprint.route('/delete', methods=['POST'])
def delete():

	#deleteDevice(request.json['id'], session['user'], session['authHeader'])
	return "SUCCESS!!"


