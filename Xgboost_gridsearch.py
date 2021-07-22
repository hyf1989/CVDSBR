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
file_label = pd.read_csv('cvdsbr_prediction_label_group.csv', index_col=0)
file_data = pd.read_csv('cvdsbr_prediction_matrix_clinical20210713.csv', index_col=0)
new_index = []
for i in range(len(file_data.index)):
    new_index.append(file_data.index[i].split('_')[0])
file_data.index = new_index
whole_train = file_label.index[file_label['label']=='train']
sub_train = file_data.index.intersection(whole_train)

X_train = file_data.loc[sub_train]
y_train = []
for i in range(len(X_train.index)):
    y_train.append(file_label.loc[X_train.index[i]]['abnormal'])
X_train = X_train.values

sss = cross_validation_split_index_list(len(y_train), 3)
X_train_1 = X_train[sss[1][0],:]
X_val_1 = X_train[sss[1][1],:]
y_train_1 = np.array(y_train)[sss[1][0]]
y_val_1 = np.array(y_train)[sss[1][1]]
print(sum(y_val_1))
#pdb.set_trace()


whole_test = file_label.index[file_label['label']=='test']
sub_test = file_data.index.intersection(whole_test)

X_test = file_data.loc[sub_test]
#pdb.set_trace()
y_test = []
for i in range(len(X_test.index)):
    y_test.append(file_label.loc[X_test.index[i]]['abnormal'])
X_test = X_test.values

lr = [0.28,0.3,0.32]
ss = [0.6,0.7,0.8]
cbt = [0.6,0.7,0.8]
cbl = [0.6,0.7,0.8]
spw = [1,1.2,1.4]
#feature selection
best_param_list = []
Val_ACC = []
Val_AUC = []
TST_ACC = []
TST_AUC = []
best_value = 0
for learning_rate in lr:
    for subsample in ss:
        for colsample_bytree in cbt:
            for colsample_bylevel in cbl:
                for scale_pos_weight in spw:
                    param_list = [learning_rate, subsample, colsample_bytree, colsample_bylevel, scale_pos_weight]
                    print(param_list)
                    model = XGBClassifier(learning_rate=learning_rate, subsample=subsample,
                                          colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                                          n_estimators=X_train.shape[1], objective='binary:logistic',
                                          scale_pos_weight=scale_pos_weight, seed=0, use_label_encoder=False,
                                          eval_metric='logloss')
                    model.fit(X_train_1, y_train_1)
                    thresholds = sort(model.feature_importances_)[::-1]
                    count = 0
                    for thresh in thresholds[0:10]:
                        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                        select_X_train = selection.transform(X_train_1)

                        # train model
                        selection_model = XGBClassifier(use_label_encoder=False,objective='binary:logistic',
                            eval_metric='logloss')
                        selection_model.fit(select_X_train, y_train_1)

                        # eval model val
                        select_X_val = selection.transform(X_val_1)
                        y_pred_val = selection_model.predict_proba(select_X_val)
                        # pdb.set_trace()
                        predictions_val = [round(value) for value in y_pred_val[:, 1]]
                        val_accuracy = accuracy_score(y_val_1, predictions_val)
                        val_auc = roc_auc_score(y_val_1, y_pred_val[:, 1])
                        count += 1
                        #print("Thresh=%.3f, n=%d, Val Acc: %.2f, Val AUC: %.2f%%" % (thresh, select_X_train.shape[1], val_accuracy * 100.0, val_auc * 100.0))

                        # eval model test
                        select_X_test = selection.transform(X_test)
                        y_pred = selection_model.predict_proba(select_X_test)
                        #pdb.set_trace()
                        predictions = [round(value) for value in y_pred[:,1]]
                        test_accuracy = accuracy_score(y_test, predictions)
                        test_auc = roc_auc_score(y_test, y_pred[:,1])
                        count += 1
                        #print("Thresh=%.3f, n=%d, test Acc: %.2f, test AUC: %.2f%%" % (thresh, select_X_train.shape[1], test_accuracy*100.0, test_auc*100.0))

                        if (best_value <= val_accuracy + val_auc):# or (val_accuracy > 0.85 and val_auc > 0.85):
                            best_param_list.append(param_list)
                            Val_ACC.append(val_accuracy)
                            Val_AUC.append(val_auc)
                            TST_ACC.append(test_accuracy)
                            TST_AUC.append(test_auc)
                            print("Excellent!")
                            print("Val Acc: %.2f, Val AUC: %.2f, Test Acc: %.2f, Test AUC: %.2f" % (
                            val_accuracy * 100.0, val_auc * 100.0, test_accuracy * 100.0, test_auc * 100.0))
                            best_value = val_accuracy + val_auc

rows = zip(best_param_list, Val_ACC, Val_AUC, TST_ACC, TST_AUC)
with open('best_param_'+'2'+'_list1.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)





