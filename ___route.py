
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

sys.path.append('/Users/syahrindradr/Documents/dzaky/VSCode/SKRIPSI/TA/feature')
from feature_extraction_glcm_bgr_hsv import testing as ts



app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class klasifikasi(Resource):
    def post(self):
        some_json = request.get_json()
        result=ts.try_classification(some_json['pathfile'])
        return {'data':result}, 201
    
api.add_resource(klasifikasi, '/klasifikasi')




if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app
