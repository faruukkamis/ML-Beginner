# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:46:40 2021

@author: faruukkamis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('ConcreteData.csv')

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(veriler.iloc[:,:-1],veriler.iloc[:,-1:],test_size=0.2, random_state=0)

# Multiple-Linear REgression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# Getting OLS Model for backward elemination if it is needed.
import statsmodels.api as sm
CCS = veriler.iloc[:,-1:].values
X_l = veriler.iloc[:,[0,1,2,3,4,5,6,7]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(CCS,X_l).fit()
print(model.summary())
# P values of all variables are very small, so no need backward elimination.

# Plotting Test values against our Predicted values to get visual comparison.
x_test = x_test.sort_index()
x_train = x_train.sort_index()
y_train = y_train.sort_index()
p1 = plt.plot(x_test.index,y_test, label='Test')
p2 = plt.plot(x_test.index,y_pred, label='Prediction')
plt.legend(loc=1)
plt.title('Multiple-Linear Regression')
plt.xlabel('index of samples')
plt.ylabel(' Concrete Compressive Strength (MPa)')









