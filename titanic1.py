# -*- coding: utf-8 -*-

# -- Sheet --

! pip install kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

!  kaggle competitions download -c titanic

! unzip "titanic.zip"

import numpy as np
import pandas as pd


df = pd.read_csv('train.csv')
df.head(5)

y = df['Survived']
data = df.drop(columns=['Survived'])

data.head(10)

data = data.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'])
data.head(10)

data.info()

data['Age'].describe()

fill_value = np.mean(data['Age'])
data['Age'] = data['Age'].fillna(fill_value)
data.head(10)

data['Embarked'].unique()

data['Embarked'].value_counts()

data['Embarked'] = data['Embarked'].fillna('S')

data.info()

sex_dict = {
    'male': 1,
    'female': 0
}
data['Sex'] = list(map(lambda x: sex_dict[x], data['Sex']))
data.head(10)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(data['Embarked'])
data['Embarked'] = le.transform(data['Embarked'])

data.head(10)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(data, y)
knn_preds = knn.predict(data)
knn_preds_proba = knn.predict_proba(data)[:, 1]

print('accuracy', accuracy_score(y, knn_preds))
print('roc_auc', roc_auc_score(y, knn_preds_proba))

knn_preds[:20]

knn_preds_proba[:20]

cross_val_scores = cross_val_score(knn, data, y, cv=5, scoring='roc_auc')
print(cross_val_scores)
print(np.mean(cross_val_scores))

lr = LogisticRegression()

lr.fit(data, y)
lr_preds = lr.predict(data)
lr_preds_proba = lr.predict_proba(data)[:, 1]

print('accuracy', accuracy_score(y, lr_preds))
print('roc_auc', roc_auc_score(y, lr_preds_proba))

cross_val_scores = cross_val_score(lr, data, y, cv=5, scoring='roc_auc')
print(cross_val_scores)
print(np.mean(cross_val_scores))

gbdt = GradientBoostingClassifier()

gbdt.fit(data, y)
gbdt_preds = gbdt.predict(data)
gbdt_preds_proba = gbdt.predict_proba(data)[:, 1]

print('accuracy', accuracy_score(y, gbdt_preds))
print('roc_auc', roc_auc_score(y, gbdt_preds_proba))

cross_val_scores = cross_val_score(gbdt, data, y, cv=5, scoring='roc_auc')
print(cross_val_scores)
print(np.mean(cross_val_scores))

gbdt = GradientBoostingClassifier()
kfold = KFold(n_splits=5, shuffle=True, random_state=123)

param_grid = {
    "max_depth": [2, 3],
    "n_estimators": [50, 100, 150]
    # "learning_rate": [0.01, 0.05, 0.1],
    # "min_child_weight":[4,5,6],
    # "subsample": [0.8, 0.9, 1]
}

CV_gbdt = GridSearchCV(estimator=gbdt, param_grid=param_grid,
                      scoring='roc_auc', cv=kfold, verbose = 1000)

CV_gbdt.fit(data, y)

data_test = pd.read_csv('test.csv')
data_test.head()

passenger_id = data_test['PassengerId']

data_test = data_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data_test.head()

data_test['Age'] = data_test['Age'].fillna(fill_value)
data_test['Embarked'] = data_test['Embarked'].fillna('S')

data_test['Fare'] = data_test['Fare'].fillna(np.mean(data['Fare']))

data_test['Sex'] = list(map(lambda x: sex_dict[x], data_test['Sex']))
data_test['Embarked'] = le.transform(data_test['Embarked'])

data_test.head()

y_pred_lr = lr.predict(data_test)
y_pred_knn = knn.predict(data_test)
y_pred_gbdt = CV_gbdt.predict(data_test)

y_pred_gbdt = pd.DataFrame(y_pred_gbdt, columns=['Survived'])
y_pred_gbdt['PassengerId'] = passenger_id
y_pred_gbdt = y_pred_gbdt[['PassengerId', 'Survived']]
y_pred_gbdt.to_csv('submission_gbdt.csv', index=None)

y_pred_gbdt.head()

