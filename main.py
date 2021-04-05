from flask import Flask, json
from flask_cors import CORS, cross_origin

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'


@api.route('/hello', methods=['GET'])
@cross_origin()
def get_companies():
    return "Hello"



api.run(host='0.0.0.0')
input("Press enter to exit")
