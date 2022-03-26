from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import json
import numpy as np

app =Flask(__name__)

loaded_model =  pickle.load(open('./model/classifyFish.sav', 'rb'))
cols = ['weight', 'length 1', 'length 2', 'length 3', 'height', 'width']

species = {0: 'Bream',
 1: 'Parkki',
 2: 'Perch',
 3: 'Pike',
 4: 'Roach',
 5: 'Smelt',
 6: 'Whitefish'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    data_unseen = np.array(int_features)
    d= data_unseen[np.newaxis,:]
    predicted_value = loaded_model.predict(data_unseen[np.newaxis,:]) 
    return render_template('results.html', species=species[predicted_value[0]])


if __name__ == '__main__':
    app.run()