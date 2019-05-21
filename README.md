# AUCROC-curve_predict_proba
USE of predict_proba

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF

X = np.array([[5,5,5,5],[10,10,10,10],[1,1,1,1],[6,6,6,6],[13,13,13,13],[2,2,2,2]])

#y = np.array([0,1,1,0,1,2])
y = np.array([0,2,1,0,1,2])

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)

# fit final model
model = RF()
model.fit(X_train,y_train)
print(model.classes_)
print(model.predict(X_test))
pred_pro = model.predict_proba(X_test)
pred_pro_class1 = model.predict_proba(X_test)[:,1]
print(pred_pro)
print(pred_pro_class1)
