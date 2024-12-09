from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

preprocessor=pickle.load(open('preprocessor .pkl','rb'))
app = Flask(__name__)                   


model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        crop = request.form['crop']
        Precipitation = float(request.form['Precipitation'])
        Humidity = float(request.form['Humidity'])
        Temperature = float(request.form['Temperature'])
        Relative = float(request.form['Relative'])

        
        features = pd.DataFrame({
            'Crop': [crop],
            'Precipitation (mm day-1)': [Precipitation],
            'Specific Humidity at 2 Meters (g/kg)': [Humidity],
            'Temperature at 2 Meters (C)': [Temperature],
            'Relative Humidity at 2 Meters (%)': [Relative],
            'Crop_Encoded': [0] 
        })

        
        transformed_features = preprocessor.transform(features)

        
        prediction = model.predict(transformed_features).reshape(1, -1)

       
        return render_template('index.html', prediction_text=f'Predicted Crop Yield: {prediction[0]}')


if __name__ == '__main__':
    app.run(debug=True)

