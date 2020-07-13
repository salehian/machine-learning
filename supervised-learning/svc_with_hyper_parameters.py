import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the data
data = pd.read_csv('./data/data01.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Split your data into train/test data with a 4-1 proportionality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# Support Vector Machine Classifier
#Hyperparameters
#C: The C parameter.
#kernel: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
#degree: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
#gamma : If the kernel is rbf, this is the gamma parameter.
model = SVC(kernel='rbf', gamma=27)

# Fit the model to the train data
model.fit(X_train,y_train)

# Make predictions on the test data		
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f'% acc)

#Plot Test and Prediction Data
fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle('Support Vector Machine Classifier', fontsize=16)
axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test)
axs[0].set_title('Test')
axs[1].scatter(X_test[:,0], X_test[:,1], c=y_pred)
axs[1].set_title('Prediction')

plt.show()