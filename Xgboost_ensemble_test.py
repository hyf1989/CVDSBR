import numpy as np
from numpy import sort
import pandas as pd
import os
import pdb
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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
import matplotlib.pyplot as plt

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

model1_clinic = XGBClassifier(learning_rate=0.28,subsample=0.7,colsample_bytree=0.8,colsample_bylevel=0.6,n_estimators=X_train_clinic.shape[1],objective='binary:logistic',scale_pos_weight=1,seed=0,use_label_encoder =False, eval_metric='logloss')
model1_clinic.fit(X_train_1_clinic, y_train_1_clinic)
thresh1_clinic = sort(model1_clinic.feature_importances_)[::-1][8]

target_columns = np.where(model1_clinic.feature_importances_>=thresh1_clinic)[0]
print('Important features of clinic are %s' % file_data_clinic.columns[target_columns].tolist())
# pdb.set_trace()
# clinic_feature_df = pd.DataFrame(model1_clinic.feature_importances_[target_columns].reshape(1,-1),columns = file_data_clinic.columns[target_columns],index=['importance'])
# clinic_feature_df.to_csv("clinic_feature_importance.csv")
# pdb.set_trace()
# p1_clinic = list(model1_clinic.feature_importances_)
# max_feature_index_list1_clinic = [*map(p1_clinic.index, heapq.nlargest(2, p1_clinic))]
#pdb.set_trace()
selection1_clinic = SelectFromModel(model1_clinic, threshold=thresh1_clinic, prefit=True)
select_X_train1_clinic = selection1_clinic.transform(X_train_1_clinic)
selection_model1_clinic = XGBClassifier(use_label_encoder =False, eval_metric='logloss',objective='binary:logistic')
selection_model1_clinic.fit(select_X_train1_clinic, y_train_1_clinic)

select_X_test1_clinic=selection1_clinic.transform(X_test_clinic)
y_pred_test1_clinic = selection_model1_clinic.predict_proba(select_X_test1_clinic)
predictions_test1_clinic = [round(value) for value in y_pred_test1_clinic[:,1]]
test_accuracy_clinic = accuracy_score(y_test_clinic, predictions_test1_clinic)
test_auc_clinic = roc_auc_score(y_test_clinic, y_pred_test1_clinic[:,1])

fpr_test,tpr_test,threshold_test = roc_curve(y_test_clinic, y_pred_test1_clinic[:,1])
lw = 6
plt.cla()
fig, ax = plt.subplots(figsize=(14,14))
plt.plot(fpr_test, tpr_test, color='red',lw=lw, label='ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.yticks(fontproperties = 'Times New Roman', size = 24)
plt.xticks(fontproperties = 'Times New Roman', size = 24)
ax.set_aspect('equal')
plt.xlabel('1 - Specificity',fontsize=24)
plt.ylabel('Sensitivity',fontsize=24)
plt.title('Clinic ROC',fontsize=30)
ss = 'AUC=%.3f, ACC=%.3f' % (test_auc_clinic,test_accuracy_clinic)

plt.text(0.75, 0, s=ss,fontdict=dict(fontsize=16, color='b',family='monospace',),bbox={'facecolor': 'blue', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0.1, #框透明度
               'pad': 0.8,#本文与框周围距离
               'boxstyle':'round'
              })
plt.savefig(r'D:\CVDSBR\result\Clinic_ROC.pdf')

test_accuracy_clinic_omics = accuracy_score(y_test_clinic[:16], predictions_test1_clinic[:16])
test_auc_clinic_omics = roc_auc_score(y_test_clinic[:16], y_pred_test1_clinic[:,1][:16])
print("Clinic Test Acc-All Data: %.2f, Clinic Test AUC-All Data: %.2f" % (test_accuracy_clinic * 100.0, test_auc_clinic * 100.0))
print("Clinic Test Acc-Only Omics Data: %.2f, Clinic Test AUC-Only Omics Data: %.2f" % (test_accuracy_clinic_omics * 100.0, test_auc_clinic_omics * 100.0))
#data preparation
file_label = pd.read_csv('cvdsbr_prediction_label_group.csv', index_col=0)
whole_train = file_label.index[file_label['label']=='train']
whole_test = file_label.index[file_label['label']=='test']
column_names = ['SM2','SP1','SP2','SP3','SP4','SP5','UM1','UP1','UP2','UP3']
# column_names = ['SP1','SP2','SP3','SP4','SP5']
# column_names = ['UP1','UP2','UP3']
# column_names = ['SM2']
# column_names = ['UM1']
results = pd.DataFrame(columns=column_names,index=whole_test)
num = 5
feature_name = []
feature_importance = []

for columnname in column_names:
    filename = [s for s in os.listdir() if columnname in s][0]
    file_data = pd.read_csv(filename, index_col=0)
    new_index = []
    for i in range(len(file_data.index)):
        new_index.append(file_data.index[i].split('_')[0])
    file_data.index = new_index
    sub_train = file_data.index.intersection(whole_train)
    sub_test = file_data.index.intersection(whole_test)

    X_train = file_data.loc[sub_train]
    y_train = []
    for i in range(len(X_train.index)):
        y_train.append(file_label.loc[X_train.index[i]]['abnormal'])
    X_train = X_train.values

    #feature selection
    model = XGBClassifier(learning_rate=0.3, subsample=0.5,reg_lambda=0, reg_alpha=0, colsample_bytree=0.5,colsample_bylevel=0.5,n_estimators=X_train.shape[1],objective='binary:logistic',scale_pos_weight=10/9,seed=0,use_label_encoder =False, eval_metric='logloss')
    model.fit(X_train, y_train)
    thresholds = sort(model.feature_importances_)[::-1]
    count = 0
    keep_features = min(max(int(2/3*np.sum(thresholds>0)),1),num)
    selection = SelectFromModel(model, threshold=thresholds[keep_features-1], prefit=True)
    select_X_train = selection.transform(X_train)
    #pdb.set_trace()
    target_columns = np.where(model.feature_importances_>=thresholds[keep_features-1])[0]
    feature_name.append(file_data.loc[sub_train].columns[target_columns].tolist())
    feature_importance.append(model.feature_importances_[target_columns].tolist())
    print('Important proteins of omics data %s are %s' % (columnname,file_data.loc[sub_train].columns[target_columns].tolist()))


    # train model
    selection_model = XGBClassifier(use_label_encoder=False,objective='binary:logistic',
        eval_metric='logloss')
    selection_model.fit(select_X_train, y_train)

    # eval model
    for test_data in whole_test:
        if test_data in file_data.index:
            X_test = file_data.loc[test_data].values.reshape(-1,file_data.shape[1])
            select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict_proba(select_X_test)
            y_pred = np.mean(y_pred,axis=0)[1]
            results[columnname][test_data] = y_pred

feature_name = [item for sublist in feature_name for item in sublist]
feature_importance = [item for sublist in feature_importance for item in sublist]

# sorted_feature_name = np.array(feature_name)[np.argsort(feature_importance)].tolist()
# sorted_feature_importance = np.sort(feature_importance)
# plt.barh(range(len(sorted_feature_name)), sorted_feature_importance,color='b',tick_label=sorted_feature_name)
# plt.tick_params(labelsize=6)
# plt.tight_layout()
# plt.xlabel('Feature Importance')
# plt.savefig(r'D:\CVDSBR\result\Figure_1_UM.pdf')
# df = pd.DataFrame(columns=sorted_feature_name,index=['Feature Importance'])
# df.loc['Feature Importance'] = sorted_feature_importance
# df.to_csv(r'D:\CVDSBR\result\Figure_1_UM.csv')

final_y_pred = np.nanmean(results[:16],axis=1)
predictions = [round(value) for value in final_y_pred]
final_y_label = file_label.loc[whole_test[:16]]['abnormal'].values
test_accuracy = accuracy_score(final_y_label, predictions)
test_auc = roc_auc_score(final_y_label, final_y_pred)
print("Omics Test Acc: %.2f, Omics Test AUC: %.2f%%" % (test_accuracy*100.0, test_auc*100.0))

#pdb.set_trace()
import copy
results_new = copy.deepcopy(results)
results_new.insert(results_new.shape[1], 'label', file_label.loc[whole_test]['abnormal'].values)
auc_all_columns = []
name_columns = results.columns.tolist()
for k in results.columns:
    results_temp = results_new[results_new[k].notna()]
    temp_label = results_temp['label'].values
    temp_score = results_temp[k].values
    auc_all_columns.append(roc_auc_score(temp_label, temp_score))

name_columns.append('all_omics')
auc_all_columns.append(test_auc)

fpr_test,tpr_test,threshold_test = roc_curve(final_y_label, final_y_pred)
lw = 6
plt.cla()
fig, ax = plt.subplots(figsize=(14,14))
plt.plot(fpr_test, tpr_test, color='red',lw=lw, label='ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.yticks(fontproperties = 'Times New Roman', size = 24)
plt.xticks(fontproperties = 'Times New Roman', size = 24)
ax.set_aspect('equal')
plt.xlabel('1 - Specificity',fontsize=24)
plt.ylabel('Sensitivity',fontsize=24)
plt.title('Omics ROC',fontsize=30)
ss = 'AUC=%.3f, ACC=%.3f' % (test_auc,test_accuracy)

plt.text(0.75, 0, s=ss,fontdict=dict(fontsize=16, color='b',family='monospace',),bbox={'facecolor': 'blue', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0.1, #框透明度
               'pad': 0.8,#本文与框周围距离
               'boxstyle':'round'
              })
plt.savefig(r'D:\CVDSBR\result\Omics_ROC.pdf')

results_all = copy.deepcopy(results)
results_all.insert(results_all.shape[1], 'clinic', y_pred_test1_clinic[:,1])
all_y_pred_omics = np.nanmean(results_all[:16],axis=1)
all_predictions_omics = [round(value) for value in all_y_pred_omics]
all_test_accuracy_omics = accuracy_score(final_y_label, all_predictions_omics)
all_test_auc_omics = roc_auc_score(final_y_label, all_y_pred_omics)
print("All Test Acc-Only Omics data: %.2f, All Test AUC-Only Omics data: %.2f%%" % (all_test_accuracy_omics*100.0, all_test_auc_omics*100.0))
name_columns.append('all_omics_with_clinic')
auc_all_columns.append(all_test_auc_omics)

df_auc = pd.DataFrame(columns = name_columns,index = ['AUC'])
df_auc.loc['AUC'] = auc_all_columns

save_name = os.path.join(r'D:\CVDSBR\result',str(num)+'_features.csv')
df_auc.to_csv(save_name)

fpr_test,tpr_test,threshold_test = roc_curve(final_y_label, all_y_pred_omics)
lw = 6
plt.cla()
fig, ax = plt.subplots(figsize=(14,14))
plt.plot(fpr_test, tpr_test, color='red',lw=lw, label='ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.yticks(fontproperties = 'Times New Roman', size = 24)
plt.xticks(fontproperties = 'Times New Roman', size = 24)
ax.set_aspect('equal')
plt.xlabel('1 - Specificity',fontsize=24)
plt.ylabel('Sensitivity',fontsize=24)
plt.title('Omics + Clinic ROC',fontsize=30)
ss = 'AUC=%.3f, ACC=%.3f' % (all_test_auc_omics,all_test_accuracy_omics)

plt.text(0.75, 0, s=ss,fontdict=dict(fontsize=16, color='b',family='monospace',),bbox={'facecolor': 'blue', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0.1, #框透明度
               'pad': 0.8,#本文与框周围距离
               'boxstyle':'round'
              })
plt.savefig(r'D:\CVDSBR\result\Omics_Clinic_ROC.pdf')


all_y_pred = np.nanmean(results_all,axis=1)
all_predictions = [round(value) for value in all_y_pred]
final_y_label_all = file_label.loc[whole_test]['abnormal'].values
all_test_accuracy = accuracy_score(final_y_label_all, all_predictions)
all_test_auc = roc_auc_score(final_y_label_all, all_y_pred)
print("All Test Acc-All data: %.2f, All Test AUC-All data: %.2f%%" % (all_test_accuracy*100.0, all_test_auc*100.0))

df_score = pd.DataFrame(columns=['predicted score'],index=file_label.loc[whole_test].index)
df_score['predicted score'] = all_y_pred
df_score.to_csv(r'D:\CVDSBR\result\score_all_omics_clinic.csv')
