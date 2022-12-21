from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

csv_data = pd.read_csv(r'competicao-um-ic/train.csv',delimiter=(','))
csv_data = csv_data[['DS_CARATER_ATENDIMENTO','DS_GRUPO','DS_TIPO_ATENDIMENTO','DS_TIPO_GUIA','DS_TIPO_ITEM','DS_TIPO_PREST_SOLICITANTE','QT_DIA_SOLICITADO','DS_STATUS_ITEM']]



csv_data['QT_DIA_SOLICITADO'] = np.where(csv_data['QT_DIA_SOLICITADO'].isnull(), 0, csv_data['QT_DIA_SOLICITADO'])
csv_data['QT_DIA_SOLICITADO'] = csv_data['QT_DIA_SOLICITADO'].astype(int)

enc = OneHotEncoder(handle_unknown='ignore')

features_values = enc.fit_transform(csv_data[['DS_CARATER_ATENDIMENTO','DS_GRUPO','DS_TIPO_ATENDIMENTO','DS_TIPO_GUIA','DS_TIPO_ITEM','DS_TIPO_PREST_SOLICITANTE']]).toarray()
catogories = enc.categories_
features_label = np.concatenate(catogories, axis=0 )


features = pd.DataFrame(features_values,columns=features_label)

csv_data = pd.concat([features,csv_data[['QT_DIA_SOLICITADO','DS_STATUS_ITEM']]],axis=1)




data_atributes = csv_data.to_numpy()[:,:43]
data_target = csv_data.to_numpy()[:,43]


X_train, X_test, y_train, y_test = train_test_split(data_atributes, data_target, test_size=0.2,random_state=109)




clf = tree.DecisionTreeClassifier()


clf = clf.fit(data_atributes, data_target)

#base teste
test_data = pd.read_csv(r'competicao-um-ic/test.csv',delimiter=(','))
test_index_column = test_data['Unnamed: 0']
test_data = test_data[['DS_CARATER_ATENDIMENTO','DS_GRUPO','DS_TIPO_ATENDIMENTO','DS_TIPO_GUIA','DS_TIPO_ITEM','DS_TIPO_PREST_SOLICITANTE','QT_DIA_SOLICITADO']]



test_data['QT_DIA_SOLICITADO'] = np.where(test_data['QT_DIA_SOLICITADO'].isnull(), 0, test_data['QT_DIA_SOLICITADO'])
test_data['QT_DIA_SOLICITADO'] = test_data['QT_DIA_SOLICITADO'].astype(int)



test_features_values = enc.fit_transform(test_data[['DS_CARATER_ATENDIMENTO','DS_GRUPO','DS_TIPO_ATENDIMENTO','DS_TIPO_GUIA','DS_TIPO_ITEM','DS_TIPO_PREST_SOLICITANTE']]).toarray()
test_catogories = enc.categories_
test_features_label = np.concatenate(test_catogories, axis=0 )


test_features = pd.DataFrame(test_features_values,columns=test_features_label)

test_data = pd.concat([test_features,test_data[['QT_DIA_SOLICITADO']]],axis=1)

test_data_atributes = test_data.to_numpy()[:,:43]


predict = clf.predict(test_data_atributes)

from itertools import zip_longest
submission = pd.DataFrame.from_records(zip_longest(test_index_column, predict), columns=['ID', 'DS_STATUS_ITEM'])

submission.set_index('ID').to_csv('submission.csv')

print("Accuracy: ",metrics.accuracy_score(test_data_target, predict))
