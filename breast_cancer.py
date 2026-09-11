from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'],columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

x = df[cancer_data['feature_names']].values
y=df['target'].values

model = LogisticRegression(solver='liblinear')
model.fit(x,y)

print(model.predict([x[0]]))
print(model.score(x,y))