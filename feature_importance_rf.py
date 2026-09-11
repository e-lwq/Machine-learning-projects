import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

x_worst=[]
for fn in cancer_data['feature_names']:
    if 'worst' in fn:
        x_worst.append(fn)
        

x=df[x_worst]
y=df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=101)

rf = RandomForestClassifier(n_estimators = 10, random_state = 10)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))

#ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
#print(ft_imp.head(10))

