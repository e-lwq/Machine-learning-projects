import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex']=='male'
x1 = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
x2 = df[['Pclass','male','Age']].values
x3 = df[['Fare','Age']].values
y = df[['Survived']].values

kf = KFold(n_splits = 5, shuffle = True)

def score_model(X,Y,kf):
    splits = kf.split(X)
    a=[]
    p=[]
    r=[]
    f=[]
    
    for train_ind, test_ind in splits:
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        X_test = X[test_ind]
        Y_test = Y[test_ind]
        
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        a.append(accuracy_score(Y_test,Y_pred))
        p.append(precision_score(Y_test,Y_pred))
        r.append(recall_score(Y_test,Y_pred))
        f.append(f1_score(Y_test,Y_pred))
        
    print('accuracy:',np.mean(a))
    print('precision:',np.mean(p))
    print('recall:',np.mean(r))
    print('f1:',np.mean(f))
    

print('training w x1')
score_model(x1,y,kf)
print('')

print('training w x2')
score_model(x2,y,kf)
print('')

print('training w x3')
score_model(x3,y,kf)

final_model = LogisticRegression(solver='liblinear')
final_model.fit(x1,y)

print(final_model.predict([[3,False,25,0,1,2]]))
print(final_model.predict([[3,True,22,1,9,7.25]]))
        