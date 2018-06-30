import os
import pandas as pd
from tqdm import *
from sklearn import preprocessing, svm, datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, precision_score,mean_absolute_error,confusion_matrix

files = []
path = "C:\Users\sternheimam\Desktop\my-notebook\user-csvs_predictions"


print "Loading..."
for filename in tqdm(os.listdir(path)):
    files.append(pd.read_csv(os.path.join(path, filename)))
	
data = pd.concat(files)
data = data.drop('Date & Time',1)
data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))
data = pd.concat([data[(data[1]==1)],data[(data[1]==0)].sample(n=len(data[(data[1]==1)]))])

X = data.drop([0,1,2], axis = 1)
y = data[1]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

dt = xgb.DMatrix(train_X.as_matrix(),label=train_y.as_matrix())
dv = xgb.DMatrix(test_X.as_matrix(),label=test_y.as_matrix())

params = {
    "eta": 0.2,
    "max_depth": 4,
    "objective": "binary:logistic",
    "silent": 1,
    "base_score": np.mean(train_y),
    'n_estimators': 1000,
    "eval_metric": "logloss"
}
model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "validation")], verbose_eval=200)

y_pred = model.predict(dv)
cm = confusion_matrix(test_y, (y_pred>0.5))
print "confusion matrix: \n",cm
TP = float(cm[1,1]) # 208 instanties zijn echt 1 en voorspeld 1 (true positive)
FP = float(cm[0,1]) # 92 instanties zijn echt 0 en voorspeld 1 (false positive)
FN = float(cm[1,0]) # 2523 instanties zijn echt 1 en voorspeld 0 (false negative)
TN = float(cm[0,0]) # 147650 instanties zijn echt 0 en voorspeld 0 (true negative)

print "TN: "+str(TN), "FP: "+str(FP)
print "FN: "+str(FN), "TP: "+str(TP)

accuracy = (TN+TP)/(TN+TP+FN+FP)
print "accuracy: "+ str(accuracy) # 0,9826214669741415

precision = TP/(TP+FP)
print "precision: "+str(precision)
# 0.693333333333: 70% of all predictions that patients will churn in 3 months are indeed true

recall = TP/(TP+FN)
print "recall: "+str(recall)
# 0.0761625778103: 7.5% of all patients who will churn in 3 months are correctly identified

f_score = (2*(precision*recall))/(precision+recall)
print "f score: "+str(f_score)