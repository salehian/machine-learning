import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the data
data = pd.read_csv('data.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Split your data into train/test data with a 4-1 proportionality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# Decision Tree Classifier
model = DecisionTreeClassifier()

## Fit the model to the train data
model.fit(X_train,y_train)

# Make predictions on the test data		
y_pred = model.predict(X_test)

# Calculate and print the accuracy
acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f'% acc)

#Plot Test and Prediction Data
fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle('DecisionTree Classifier', fontsize=16)
axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test)
axs[0].set_title('Test')
axs[1].scatter(X_test[:,0], X_test[:,1], c=y_pred)
axs[1].set_title('Prediction')

plt.show()