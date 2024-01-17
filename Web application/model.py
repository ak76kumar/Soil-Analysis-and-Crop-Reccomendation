import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('crop_recommendation.csv')
X = np.array(df.iloc[:, 0:7])
y = np.array(df.iloc[:, 7:])

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y.reshape(-1))


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
model = RF.fit(Xtrain, Ytrain)

predicted_values = RF.predict(Xtest)
print(predicted_values)

from sklearn import metrics
print(metrics.accuracy_score(Ytest, predicted_values))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))