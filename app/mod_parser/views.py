from flask import Blueprint, render_template, request, redirect, url_for

parser_blueprint = Blueprint('parser', __name__)

@parser_blueprint.route('/parser')
def ingest():
    return 'Parser!'
