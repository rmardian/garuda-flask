from flask import Blueprint

correlation_blueprint = Blueprint('correlation', __name__)

@correlation_blueprint.route('/correlation')
def ingest():
    return render_template('correlation/index.html')

