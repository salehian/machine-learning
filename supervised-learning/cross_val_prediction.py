import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


# Load the boston dataset
boston = datasets.load_boston()
y = boston.target
X = boston.data

# Linear Regression
model = LinearRegression()

# Prediction obtained by cross validation
y_pred = cross_val_predict(model, X, y, cv=10)

# Plot
fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
