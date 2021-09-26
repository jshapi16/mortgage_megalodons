import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.models import load_model
import h5py


app = Flask(__name__)
model = load_model('mortgage_deep_learning17_3.h5')

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title='Home Page')

@app.route("/model")
def model():
    return render_template('model.html', title='Our Model')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/prediction")
def prediction():
    return render_template('prediction.html', title='Prediction')

@app.route('/prediction',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('prediction.html', title='Prediction')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html', title='Dashboard')

@app.route('/dashboard',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
