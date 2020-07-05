import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the boston dataset
X, y = datasets.load_boston(return_X_y=True)

# Use only one feature
X = X[:, np.newaxis, 2]

# Split your data into train/test data with a 4-1 proportionality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=89)

# Linear Regression Classifier
model = LinearRegression()

# Fit the model to the train data
model.fit(X_train,y_train)

# Make predictions on the test data		
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: %.2f'% mse)

# Plot Linear Regression
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, y_pred, color='green', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
