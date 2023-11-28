from flask import Flask, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


import sys
sys.path.append('/Users/syahrindradr/Documents/dzaky/VSCode/SKRIPSI/TA/feature')
from feature_extraction_glcm_bgr_hsv import testing as ts

@app.route('/klasifikasi', methods=['POST'])
@cross_origin()
def helloWorld():
    some_json = request.get_json()
    string_url="/Applications/XAMPP/xamppfiles/htdocs/cauliflower/asset/"+some_json['pathfile']
    result=ts.try_classification(string_url)
    list_data=result[3].tolist()
    return {
                'hasil':result[0],
                'path_remove':result[1],
                'path_hsv':result[2],
                'path_mentah':'asset/'+some_json['pathfile'],
                'array_data':list_data
            }, '200 OK'

if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app
