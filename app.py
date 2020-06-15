from __future__ import division, print_function
# coding=utf-8
import numpy as np
import sys
import os
import glob
import re
from flask import Flask, request, render_template
import pickle

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
model_diabetes = pickle.load(open('modeldiabetes.pkl', 'rb'))
model_heart = pickle.load(open('modelheart.pkl', 'rb'))
MODEL_PATH = 'models/pneumonia_vgg16.h5'
# Load your trained model custom_objects={'Adam':lambda **kwargs : hvd.DistributedOptimizer(keras.optimizers.Adam(**kwargs))}
model = load_model("pneumonia_vgg16.h5",compile=False)
model._make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/')
def home():
    return render_template('main.html')
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")
@app.route("/predictpage/")
def predictpage():
    return render_template("predictpage.html")
@app.route("/utilize/")
def utilize():
    return render_template("utilize.html")
@app.route("/contacts/")
def contacts():
    return render_template("contacts.html")
@app.route("/heart")
def heart():
    return render_template("heart.html")
@app.route("/predictheart",methods=["POST"])
def predictheart():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_heart.predict_proba(final_features)
    pofnotdis=(prediction[0][0])*100
    pofhavedis=(prediction[0][1])*100
    preddict=[" Negative"," Positive"]
    finalres=preddict[np.argmax(prediction)]

    #output = int(prediction[0])
    output="Percentage of Not Having Heart Disease : {}% \n Percentage Of Having Heart Disease : {}% \n Result : {}".format(pofnotdis,pofhavedis,finalres)

    return render_template('heart.html', prediction_text='{}'.format(output))

    

@app.route('/predictdiabetes',methods=['POST'])
def predictdiabetes():
    '''
    For rendering results on HTML GUI
    '''
    pregnancies=float(request.form["Pregnancies"])
    Glucose=float(request.form["Glucose"])
    BloodPressure=float(request.form["BloodPressure"])
    SkinThickness=float(request.form["SkinThickness"])
    Insulin=float(request.form["Insulin"])
    Weight=float(request.form["Weight"])
    Height=float(request.form["Height"])
    PedigreeFunction=float(request.form["PedigreeFunction"])
    Age=float(request.form["Age"])

    #calculation of bmi
    BMI=float(Weight/(Height**2))

    float_features=[pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,PedigreeFunction,Age]
    final_features=[np.array(float_features)]
    prediction=model_diabetes.predict_proba(final_features)
    
    #float_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(float_features)]
    #prediction = model.predict(final_features)
    

    predictiondict={'0':"No Chances to have Diabetes",
                    '1':"Chances to have Diabetes"}
    notdiabetes=prediction[0][0]
    diabetes=prediction[0][1]
    numberprediction=model_diabetes.predict(final_features)
    finaldiabetesresult="Probability of Having Diabetes : {}                  Probability of Not Having diabetes : {}          Final Result : {}".format(diabetes*100,notdiabetes*100,predictiondict[str(numberprediction[0])]) 

    #output = str(prediction[0])

    return render_template('diabetes.html', prediction_text='{}'.format(finaldiabetesresult))

@app.route('/pneumonia', methods=['GET'])
def pneumonia():
    # Main page
    return render_template('pneumonia.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        classes = model_predict(file_path, model)
        names=['NORMAL','PNEUMONIA']
        ConfidenceofFalsity=classes[0][0]
        ConfidenceofPositivity=classes[0][1]
        preddict={0:'NORMAL',1:'PNEUMONIA'}
        Result=preddict[np.argmax(classes)]
        output="Falsity Confidence : {}% \n Positivity Confidence : {}% \n Result : {}".format(ConfidenceofFalsity*100,ConfidenceofPositivity*100,Result)

        #result=names[np.argmax(preds)]
        return output #result
    return None






if __name__ == "__main__":
    app.run(debug=True)
