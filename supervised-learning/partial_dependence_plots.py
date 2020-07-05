import pandas as pd
from numpy.random import rand
from sklearn.ensemble import RandomForestRegressor
from pdpbox import pdp
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt

# Number of samples
n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

# Create y
y = X1*X2

# Sreate dataframe from X1, X2, y
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = df.drop(['y'], axis=1)

model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, df.y)

# Calculate and show partial dependence plot
pdp_dist = pdp.pdp_isolate(model=model, dataset=df, model_features=['X1', 'X2'], feature='X1')
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()

perm = PermutationImportance(model).fit(predictors_df, df.y)

# Show the weights for the permutation importance
eli5.show_weights(perm, feature_names = ['X1', 'X2'])