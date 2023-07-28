import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Dataset Import
df = pd.read_csv('drug200.csv')


# EDA & Preproccessing
print(df.info())

df['Na_to_K'] = df['Na_to_K'].round()

print(df['Na_to_K'])

for c in df.columns:
    if df[c].dtype == 'object':
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

print(df.dtypes)


# Train Test Split
features = df.drop('Drug', axis= 1)
target = df['Drug']

print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.2, random_state= 42, shuffle= True)


# Model Training

models = [DecisionTreeClassifier(), RidgeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]

for m in models:
    print(f'{m}')

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy : \n{accuracy_score(Y_train, pred_train)}')
    
    pred_test = m.predict(X_test)
    print(f'Test Accuracy : \n{accuracy_score(Y_test, pred_test)}\n')