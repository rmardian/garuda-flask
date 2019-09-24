from flask import Blueprint, render_template, request, redirect, url_for, session

deploy_blueprint = Blueprint('deploy', __name__)

from app.mod_deploy.controllers import moveModel, writeDB, clearTemp

@deploy_blueprint.route('/')
def index():
	if 'user' in session and session['logged_in'] == True:
	    return redirect(url_for('train'))
	return redirect(url_for('auth.login'))

@deploy_blueprint.route('/deploy', methods=['GET', 'POST'])
def deploy():

	if request.method == 'POST':

		payload = {}

		payload['deploy'] = request.form.getlist('deploy')
		payload['cols'] = request.form.get('cols')
		payload['vals'] = request.form.get('vals')
		payload['owner'] = request.form.get('owner')
		payload['timestamp'] = request.form.get('timestamp')
		payload['cv'] = request.form.get('cv')
		payload['hy'] = request.form.get('hy')
		payload['description'] = 'A simple machine learning model deployment example.'

		#try:
		moveModel(payload['deploy'])
		clearTemp()
		writeDB(payload)
		#except:
		#	return render_template('page_500.html')

		return render_template('train.html', message="Model(s) successfully deployed!")


	if 'user' in session and session['logged_in'] == True:
		return redirect(url_for('train'))

	return redirect(url_for('auth.login'))
