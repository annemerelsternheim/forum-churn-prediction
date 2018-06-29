######################################################################
# IMPORTS
######################################################################
import os # for path
import pandas as pd # for dataframes
import xgboost as xgb # for the XGBoost classifier
from sklearn import preprocessing # for minmaxscaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold # for doing grid search, for making train and test sets from X and y, for doing 10-fold cross validation
#from sklearn.cross_validation import StratifiedKFold # for doing (stratified) k-fold. This is an old module which raises a deprecationwarning.
from sklearn.metrics import roc_auc_score # for determining the predictive accuracy of the model
import cPickle as pickle
import json


######################################################################
# FUNCTION DEFINITIONS
######################################################################

def Get_data(path,files=[]):
	""" function needs a path to a folder where all csv-files-to-be-concatenated are placed, and returns a Pandas data frame """
	print "Getting data..."
	for filename in os.listdir(path):
		files.append(pd.read_csv(os.path.join(path, filename)))
	data=pd.concat(files)
	data = data.drop('Date & Time',axis=1)
	data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))
	shuffled_data = data.sample(frac=1)
	return shuffled_data
	
def Balance_Data(data,dependent):
	""" function needs a Pandas data frame, selects only the positive values from the dependent variable, and selects as many negatives.
	Returns the concatenated data frame of these """
	#print "Balancing data..."
	positives = data[(data[dependent]==1)]
	negatives = data.drop(positives.index).sample(n=len(positives))
	balanced_data = pd.concat([positives,negatives])
	return balanced_data
	
def Shrink_data(data,dependent):
	""" This function takes a data set and returns a smaller version of it, preserving the 1:0 ratio """
	print "Shrinking data..."
	positives = data[(data[dependent]==1)].sample(frac=0.05)
	negatives = data.drop(positives.index).sample(frac=0.05)
	shrunk_data = pd.concat([positives,negatives])
	return shrunk_data
	
######################################################################
# MAIN
######################################################################

path_in = "./csvs"
path_out = "./output"

data = Balance_Data(Get_data(path_in),2) # all files

y = data[2]
X = data.drop([0,1,2],axis=1)
	
# grid search
model = xgb.XGBClassifier(seed=42)
learning_rate = [0.05, 0.1, 0.2]
n_estimators = [800]
max_depth = [8,10,12]
reg_lambda = [0.5,5,50,500]
reg_alpha = [0.5,5,50,500]
gamma = [0,0.01,0.1,1,5]
min_child_weight = [1,2,4,8]



param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth, reg_alpha=reg_alpha)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))