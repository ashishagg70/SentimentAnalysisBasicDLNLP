import os

from flask import Flask
from flask_cors import CORS

def create_app(test_config=None):
    '''
    Creates FLASK app with required configurations and route bindings
    '''
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    
    from .ratings import bp
    app.register_blueprint(bp,url_prefix='/ratings')

    from .review_processing import load_vocab,load_stop_words
    load_vocab()
    load_stop_words()
    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app