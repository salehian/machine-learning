import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the boston dataset
X, y = datasets.load_boston(return_X_y=True)

# Use only one feature
X = X[:, np.newaxis, 2]


# Split your data into train/test data with a 4-1 proportionality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=1, random_state=89)

## Fit the model to the train data
model.fit(X_train, y_train)

# Make predictions on the test data		
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: %.2f'% mse)