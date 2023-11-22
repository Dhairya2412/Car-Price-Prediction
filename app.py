from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

# loading the label encoder 
#le=pickle.load(open('label_encoder.pkl','rb'))

# loading my mlr model
model=pickle.load(open('model.pkl','rb'))

#loading Scaler
scalar=pickle.load(open('scaler.pkl','rb'))

# Flask is used for creating your application
# render template is use for rendering the html page


app= Flask(__name__)  # your application


@app.route('/')  # default route 
def home():
    return render_template('home.html') # rendering if your home page.

@app.route('/pred',methods=['POST']) # prediction route
def predict1():
    '''
    For rendering results on HTML 
    '''
    
    rd = request.form["User ID"]
    ad= request.form["Age"]
    ms = request.form["Annual Salary"]
    s = request.form["Gender"]
    t =  [[float(rd),float(ad),float(ms),float(s)]] 
    X=scalar.transform(t)
    output =model.predict(X)
    print(output)
    
    
    return render_template("home.html", result = "The predicted Price  is  "+str(np.round(output[0])))
    
    
    
# running your application
if __name__ == "__main__":
    app.run()