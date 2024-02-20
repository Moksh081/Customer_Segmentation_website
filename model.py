import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

np.random.seed(42)

data = {'AnnualIncome': np.random.randint(30000, 100000, 100),
        'SpendingScore': np.random.randint(1, 100, 100)}
df = pd.DataFrame(data)

X = df.values
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Saving the model
pickle.dump(kmeans, open('model.pkl', 'wb'))
