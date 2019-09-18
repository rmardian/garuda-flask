from flask import Blueprint, render_template, request

denormalization_blueprint = Blueprint('denormalization', __name__)

@denormalization_blueprint.route('/')
def index():
    return render_template('denormalization/index.html')

@denormalization_blueprint.route('/keys', methods=['POST', 'GET'])
def keys():

    if request.method == "POST":

        return render_template('denormalization/output.html')

    return render_template('denormalization/input.html')


