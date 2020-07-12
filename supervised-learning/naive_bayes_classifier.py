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