import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

# Read the data
data = pd.read_csv('data.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# Tuning Parameters
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# Create a scorer
scorer = make_scorer(f1_score)

# Support Vector Machine Classifier
model = SVC(random_state=89)

# Create the object
grid_obj = GridSearchCV(model, parameters, scoring=scorer)

# Fit the data
grid_fit = grid_obj.fit(X_train, y_train)

print('grid_fit.best_estimator_: ', grid_fit.best_estimator_)
print('grid_fit.best_score_: ', grid_fit.best_score_)
print('grid_fit.best_params_: ', grid_fit.best_params_)