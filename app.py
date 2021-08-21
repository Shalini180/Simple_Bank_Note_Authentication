"""
Created on Fri Aug 13 09:10:39 2021

@author: SHALI
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

   
@app.route("/")
def welcome():
    return "Welcome All"
@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    """ Authentification of banks Note
    ---
    parameters:
       -name:variance
        in:query
        type:number
        required:true
       -name:skewness
        in:query
        type:number
        required:true
       -name:curtosis
        in:query
        type:number
        required:true
       -name:entropy
        in:query
        type:number
        required:true
    responses:
        200:
            description:The output values
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is " + str(prediction)
@app.route('/predict_file',methods=["POST"])
def predict_note_file():
        """ Authentification of banks Note
    ---
    parameters:
       -name:file
        in:formData
        type:file
        required:true
    responses:
        200:
            description:The output values
    """
        df_test=pd.read_csv(request.files.get("file"))
        classifier.predict(df_test)
        prediction=classifier.predict(df_test)
        return "The predicted values is " + str(list(prediction))

if __name__=='__main__':
    app.run()