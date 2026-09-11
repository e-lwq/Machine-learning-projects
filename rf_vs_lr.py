from sklearn.datasets import make_circles
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

x,y = make_circles(noise=0.2,factor=0.5,random_state=1)

kf = KFold(n_splits=5, shuffle=True, random_state=1)

lr_score=[]
rf_score=[]

for train_ind, test_ind in kf.split(x):
    x_train, y_train = x[train_ind], y[train_ind]
    x_test, y_test = x[test_ind], y[test_ind]
    
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(x_train,y_train)
    lr_score.append(lr.score(x_test,y_test))
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x_train,y_train)
    rf_score.append(rf.score(x_test,y_test))

print(np.mean(lr_score))
print(np.mean(rf_score))