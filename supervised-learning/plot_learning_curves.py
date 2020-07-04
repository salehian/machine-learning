import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

# Read the data
data = pd.read_csv('data.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(89)

# Leave only one of the estimator lines below uncommented to see the learning curve

# Logistic Regression
estimator = LogisticRegression()

# Gradient Boosting Classifier
#estimator = GradientBoostingClassifier()

# Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=999')

# Randomize data before plotting Learning Curves
permutation = np.random.permutation(y.shape[0])
X2 = X[permutation,:]
y2 = y[permutation]

train_sizes, train_scores, test_scores = learning_curve(
	estimator, X2, y2, cv=None, n_jobs=None, train_sizes = np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()
plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")

plt.plot(train_scores_mean, 'o-', color="g",
         label="Training score")
plt.plot(test_scores_mean, 'o-', color="b",
         label="Cross-validation score")
plt.legend(loc="best")
plt.show()
