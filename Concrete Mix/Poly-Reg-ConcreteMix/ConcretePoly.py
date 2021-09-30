# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:50:13 2021

@author: faruukkamis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('ConcreteData.csv')

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(veriler.iloc[:,:-1],veriler.iloc[:,-1:],test_size=0.2, random_state=0)

# Polynomial Regression power of 2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y_train)
y_pred = lin_reg2.predict(poly_reg.fit_transform(x_test))

# Plotting Test values against our Predicted values to get visual comparison.
x_test = x_test.sort_index()
x_train = x_train.sort_index()
y_train = y_train.sort_index()
p1 = plt.plot(x_test.index,y_test, label='Test')
p2 = plt.plot(x_test.index,y_pred, label='Prediction')
plt.legend(loc=1)
plt.title('Polynomial Regression')
plt.xlabel('index of samples')
plt.ylabel(' Concrete Compressive Strength (MPa)')

