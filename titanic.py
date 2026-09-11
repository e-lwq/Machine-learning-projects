import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex'] == 'male'

model = LogisticRegression()

x = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values

model.fit(x,y)

y_pred = model.predict(x)
print((y==y_pred).sum()/y.shape[0])
print(sklearn.metrics.accuracy_score(y,y_pred))

'''
a = model.coef_[0][0]
b = model.coef_[0][1]
c = model.intercept_[0]

print(a,b,c)
plt.scatter(df['Fare'],df['Age'],c=df['Survived'])
plt.plot([0,-c/b],[-c/a,0])
plt.xlabel('Fare')
plt.ylabel('Age')'''
