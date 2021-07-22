import numpy as np
from numpy import sort
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import os
import pickle
import random
import argparse
import csv
import heapq
from sklearn.model_selection import KFold # import KFold

def cross_validation_split_index_list(n,k):
    random.seed(1)
    index = np.arange(n)
    random.shuffle(index)
    list = []
    first_index = 0
    step = round(n/k)
    for i in range(k):
        sub_list = []
        if i != k-1:
            test_temp_index = index[first_index:(first_index+step)]
            diff = set(index).difference(set(test_temp_index))
            train_temp_index = []
            for j in diff:
                train_temp_index.append(j)
        else:
            test_temp_index = index[first_index:]
            diff = set(index).difference(set(test_temp_index))
            train_temp_index = []
            for j in diff:
                train_temp_index.append(j)
        sub_list.append(train_temp_index)
        sub_list.append(test_temp_index)
        list.append(sub_list)
        first_index += step
    return list
#data preparation
file_label_clinic = pd.read_csv('cvdsbr_prediction_label_group.csv', index_col=0)
#file_data = pd.read_csv('cvdsbr_prediction_matrix_UP1.csv', index_col=0)
file_data_clinic = pd.read_csv('cvdsbr_prediction_matrix_clinical20210713.csv', index_col=0)
new_index_clinic = []
for i in range(len(file_data_clinic.index)):
    new_index_clinic.append(file_data_clinic.index[i].split('_')[0])
file_data_clinic.index = new_index_clinic
whole_train_clinic = file_label_clinic.index[file_label_clinic['label']=='train']
sub_train_clinic = file_data_clinic.index.intersection(whole_train_clinic)

X_train_clinic = file_data_clinic.loc[sub_train_clinic]
y_train_clinic = []
for i in range(len(X_train_clinic.index)):
    y_train_clinic.append(file_label_clinic.loc[X_train_clinic.index[i]]['abnormal'])
X_train_clinic = X_train_clinic.values

sss_clinic = cross_validation_split_index_list(len(y_train_clinic), 3)
X_train_1_clinic = X_train_clinic[sss_clinic[1][0],:]
X_val_1_clinic = X_train_clinic[sss_clinic[1][1],:]
y_train_1_clinic = np.array(y_train_clinic)[sss_clinic[1][0]]
y_val_1_clinic = np.array(y_train_clinic)[sss_clinic[1][1]]
print(sum(y_val_1_clinic))

whole_test_clinic = file_label_clinic.index[file_label_clinic['label']=='test']
sub_test_clinic = file_data_clinic.index.intersection(whole_test_clinic)

X_test_clinic = file_data_clinic.loc[sub_test_clinic]
y_test_clinic = []
for i in range(len(X_test_clinic.index)):
    y_test_clinic.append(file_label_clinic.loc[X_test_clinic.index[i]]['abnormal'])
X_test_clinic = X_test_clinic.values
#pdb.set_trace()

model1_clinic = XGBClassifier(learning_rate=0.28,subsample=0.7,colsample_bytree=0.8,colsample_bylevel=0.6,n_estimators=X_train_clinic.shape[1],objective='binary:logistic',scale_pos_weight=1,seed=0,use_label_encoder =False, eval_metric='logloss')
model1_clinic.fit(X_train_1_clinic, y_train_1_clinic)
thresh1_clinic = sort(model1_clinic.feature_importances_)[::-1][8]
p1_clinic = list(model1_clinic.feature_importances_)
max_feature_index_list1_clinic = [*map(p1_clinic.index, heapq.nlargest(2, p1_clinic))]
#pdb.set_trace()
selection1_clinic = SelectFromModel(model1_clinic, threshold=thresh1_clinic, prefit=True)
select_X_train1_clinic = selection1_clinic.transform(X_train_1_clinic)
selection_model1_clinic = XGBClassifier(use_label_encoder =False, eval_metric='logloss',objective='binary:logistic')
selection_model1_clinic.fit(select_X_train1_clinic, y_train_1_clinic)
select_X_val1_clinic = selection1_clinic.transform(X_val_1_clinic)
y_pred1_clinic = selection_model1_clinic.predict_proba(select_X_val1_clinic)
predictions1_clinic = [round(value) for value in y_pred1_clinic[:,1]]
val_accuracy_clinic = accuracy_score(y_val_1_clinic, predictions1_clinic)
val_auc_clinic = roc_auc_score(y_val_1_clinic, y_pred1_clinic[:,1])
select_X_test1_clinic=selection1_clinic.transform(X_test_clinic)
y_pred_test1_clinic = selection_model1_clinic.predict_proba(select_X_test1_clinic)
predictions_test1_clinic = [round(value) for value in y_pred_test1_clinic[:,1]]
test_accuracy_clinic = accuracy_score(y_test_clinic, predictions_test1_clinic)
test_auc_clinic = roc_auc_score(y_test_clinic, y_pred_test1_clinic[:,1])
print(select_X_train1_clinic.shape)
print("Val Acc: %.2f, Val AUC: %.2f, Test Acc: %.2f, Test AUC: %.2f" % (val_accuracy_clinic * 100.0, val_auc_clinic * 100.0, test_accuracy_clinic * 100.0, test_auc_clinic * 100.0))





