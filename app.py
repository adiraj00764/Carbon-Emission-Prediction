import flask
from flask import request
import pandas as pd
import numpy as np
from numpy import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def drop_duplicate(data, subset):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    # subset is list where you have to put all column for duplicate check
    data.drop_duplicates(subset, keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before-after)


df = pd.read_excel(
    "https://github.com/adiraj00764/Tractor-Supply/blob/main/TrailersReceivedByDC.xlsx?raw=true")

X = df[['Scrub_dc_no', 'Year', 'Week']]
y = df['Inbound Truckloads']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# create regressor object
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
# fit the regressor with x and y data
regressor_rf.fit(X, y)

# RU
df2 = pd.read_excel(
    "https://github.com/adiraj00764/Tractor-Supply/blob/main/ReceiptUnitsByCategory.xlsx?raw=true")
X2 = df2[['SCRUB_DC_NO', 'SCRUB_Category_no', 'Year', 'Week']]
y2 = df2['ReceiptUnits']
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=43)
subset = ['SCRUB_DC_NO', 'ReceiptUnits']
drop_duplicate(df2, subset=subset)

grad_boost = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=500, random_state=42, max_features=4, alpha=0.1, max_depth=5)
grad_boost.fit(X2_train, y2_train)


def makePredictionRU(dc_no, year, week, category_no):
    SCRUB_DC_NO = dc_no
    SCRUB_Category_no = category_no
    Year = year
    Week = week
    prediction = grad_boost.predict(
        [[SCRUB_DC_NO, SCRUB_Category_no, Week,Year]])
    return(prediction[0])


def makePrediction(dc_no, year, week):
    prediction = regressor_rf.predict([[dc_no, year, week]])
    return(prediction[0])


# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the route

# inbt


@app.route('/inbt', methods=['GET', 'POST'])
def func_inbt():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Extract the input
        Scrub_dc_no = flask.request.form['Scrub_dc_no']
        Year = flask.request.form['Year']
        Week = flask.request.form['Week']
        # Get the model's prediction
        prediction = makePrediction(Scrub_dc_no, Year, Week)
        predict = prediction.round()
        predict = int(predict)

        # # Render the form again, but add in the prediction and remind user
        # # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Scrub_dc_no': Scrub_dc_no,
                                                     'Year': Year,
                                                     'Week': Week},
                                     result=predict,
                                     )


@app.route('/ru', methods=['GET', 'POST'])
def func_ru():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main2.html'))

    if flask.request.method == 'POST':
        # Extract the input
        Scrub_dc_no = flask.request.form['Scrub_dc_no']
        Year = flask.request.form['Year']
        Week = flask.request.form['Week']
        Category = flask.request.form['Category']
        # Get the model's prediction
        prediction = makePredictionRU(Scrub_dc_no, Year, Week, Category)
        predict = prediction.round()
        predict = int(predict)

        # # Render the form again, but add in the prediction and remind user
        # # of the values they input before
        return flask.render_template('main2.html',
                                     original_input={'Scrub_dc_no': Scrub_dc_no,
                                                     'Year': Year,
                                                     'Week': Week,
                                                     'Category': Category},
                                     result=predict,
                                     )


@app.route('/', methods=['GET'])
def main():
    return(flask.render_template('options.html'))


if __name__ == '__main__':
    app.run()
