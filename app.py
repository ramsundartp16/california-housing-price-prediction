import pickle
from flask import Flask,request,jsonify,render_template,url_for,flash,app
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




