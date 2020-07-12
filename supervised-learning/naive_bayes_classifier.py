import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the data
# Dataset from - https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
data = pd.read_table('./data/SMSSpamCollection',
					  sep='\t',
					  header=None,
					  names=['label', 'sms_message'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['sms_message'], 
                                                    data['label'], 
                                                    random_state=89)

# Print the first 5 rows
print(data.head())

# Line Separator
line = 33*"="
print(line.center(80))
print('Number of rows in the data set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Create bag of words
count_vector = CountVectorizer()

# Fit the training data
training_data = count_vector.fit_transform(X_train)

# Transform test data
testing_data = count_vector.transform(X_test)

# Train Multinomial Naive Bayes Classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)