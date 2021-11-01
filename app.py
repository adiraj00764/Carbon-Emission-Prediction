from flask import Flask, request, render_template

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('co-emissions-per-capita.csv')

X = df[['Code','Year']]
y = df['Annual CO2 emissions (per capita)']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state = 0)

regressor_rf = RandomForestRegressor(n_estimators=100,random_state=0)
regressor_rf.fit(X,y)
rf = regressor_rf.predict(X_test)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/code')
def code():
    return render_template('code.html')

@app.route('/predict',methods=['POST'])
def predict():
    Year = request.form['year']
    Code = request.form['code']
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressor_rf.predict(final_features)

    output = round(prediction[0], 2)
    output1 = round(prediction[0],0)

    return render_template('index.html', prediction_text='Carbon Emission for {0} Per Capita is {1}'.format(Year,output),prediction_text1='You have to plant at least {0} 🌳 and keep them alive for next 25 Years to counter predicted carbon emission '.format(output1*4))

if __name__ == "__main__":
    app.run(debug=True)