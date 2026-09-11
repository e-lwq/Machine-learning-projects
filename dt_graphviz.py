from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import graphviz
from IPython.display import Image

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex']=='male'
feature_names = ['Pclass','male']

X = df[feature_names].values
Y = df[['Survived']].values

dt = DecisionTreeClassifier()
dt.fit(X,Y)

dot_file = export_graphviz(dt, feature_names = feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename = 'tree', format='png',cleanup=True)
