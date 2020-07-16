import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.cluster import KMeans

# Load the wholesale customers dataset
data = pd.read_csv("./data/customers.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
print ("Wholesale customers dataset has {} samples with {} features each".format(*data.shape))

display(data.describe())