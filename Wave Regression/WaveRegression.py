# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:06:27 2021

@author: faruukkamis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('CoastalData.csv')

# Clean up messy data.
data.drop(data[data['Hs'] == -99.9].index, inplace = True)
data.drop(data[data['Peak Direction']== -99.9].index, inplace = True)

# Separating independent and dependent variables.
X = data.iloc[:,[3,4,5,6]].values       # Tz,Tp,Peak Direction,SST
Y = data.iloc[:,1:3].values             # Hs,Hmax

# Train - Test Split.
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Multiple Linear Regression.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# Visualizing results to get comparison.
# x = y line is the line where prediction is equal to test data.
plt.title('Multiple-Linear Regression')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.scatter(y_test[:,0:1],y_pred[:,0:1])
plt.show()

# Random Forest Regression.
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y)
y_pred1 = rf_reg.predict(x_test)

plt.title('Random Forest Regression')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.scatter(y_test[:,0:1],y_pred1[:,0:1])
plt.show()

# Random Forest Algarith have more accurate predictions than M-Linear Algorithm on this data.

from sklearn.metrics import r2_score
print('R^2 value of Multiple Linear Regression: ',r2_score(y_test, y_pred))
print('R^2 value of Random Forest Regression: ', r2_score(y_test, y_pred1))
