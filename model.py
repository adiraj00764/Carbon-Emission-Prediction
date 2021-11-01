import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('co-emissions-per-capita.csv')

X = df[['Code','Year']]
y = df['Annual CO2 emissions (per capita)']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state = 0)

regressor_rf = RandomForestRegressor(n_estimators=100,random_state=0)
regressor_rf.fit(X,y)
rf = regressor_rf.predict(X_test)


# Saving model to disk
#pickle.dump(rf, open('emission.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('emission.pkl','rb'))
#print(model.predict([[2, 2030]]))