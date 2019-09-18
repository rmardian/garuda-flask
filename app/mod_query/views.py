from flask import Blueprint, render_template, request, redirect, url_for

query_blueprint = Blueprint('query', __name__)

@query_blueprint.route('/query')
def ingest():
    return 'success!'
    #return render_template('load/index.html')

'''
@load_blueprint.route('/form', methods=['POST', 'GET'])
def ingest():
    
    if request.method == "POST":

        name = request.form['name']
        description = request.form['description']

        user = User(name, description)
        x = collection.insert_one(user.__dict__)

        return redirect(url_for('load.ingested',
                                name=name,
                                description=description))
    
    return render_template('load/input.html')

@load_blueprint.route('/show', methods=['GET'])
def ingested():
    name = request.args.get('name')
    description = request.args.get('description')
    return render_template("load/output.html",
                           name=name,
                           description=description)
'''


