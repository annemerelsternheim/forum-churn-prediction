######################################################################
# IMPORTS
######################################################################
import os # for path
import pandas as pd # for dataframes
import xgboost as xgb # for the XGBoost classifier
from sklearn import preprocessing # for minmaxscaler
from itertools import combinations # for all combination possibilities from list items
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit, StratifiedKFold # for doing grid search, for making train and test sets from X and y, for doing 10-fold cross validation
#from sklearn.cross_validation import StratifiedKFold # for doing (stratified) k-fold. This is an old module which raises a deprecationwarning.
from sklearn.metrics import roc_auc_score, f1_score # for determining the predictive accuracy of the model
from collections import defaultdict
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
	
def Balance_Data(data, dependent):
	""" function needs a Pandas data frame, selects only the positive values from the dependent variable, and selects as many negatives.
	Returns the concatenated data frame of these """
	#print "Balancing data..."
	positives = data[(data[dependent]==1)]
	negatives = data.drop(positives.index).sample(n=len(positives))
	balanced_data = pd.concat([positives,negatives])
	return balanced_data
	
def keywithmaxval(dictionary):
	""" a) create a list of the dict's keys and values; 
	 b) return the key with the max value"""  
	v=list(dictionary.values())
	k=list(dictionary.keys())
	return k[v.index(max(v))]
	 
def f1(preds, dtrain):
	labels = dtrain.get_label() 
	binpreds = [1. if y_cont > .5 else 0. for y_cont in preds]
	return "f1", f1_score(labels, binpreds)
	
######################################################################
# MAIN
######################################################################

path_in = "./csvs"
path_out = "./output"

feature_groups = [(3,11),(13,14),(6,7),(9,10,12),(4,5,8)] # inactivity, opinionmining past, opinionmining, textual past, textual
dependent_variables = [0,1,2]

# get the data from the files
data = Get_data(path_in) # all files

variable_combinations_results_balanced = defaultdict(list)
variable_combinations_results_unbalanced = defaultdict(list)

with open(os.path.join(path_out,'COPY_FIRST_if_code_is_running8.txt'),"w") as interim:
	interim.write("B/U;dependent;independent;AUC;eta;depth;f1\n")
	
	folds = 10
	bounds = [int(s*(len(data)/folds)) for s in range(folds+1)]
	for i,b in enumerate(bounds):
		if bounds[i]!=bounds[-1]:
			print "FOLD %i" %i
			
			fold_data = data[bounds[i]:bounds[i+1]]
			# you are within one of the folds right now.
			# split this fold into three parts: train development test
			train_data = fold_data.sample(frac=0.6)
			rest = fold_data.drop(train_data.index)
			development_data = rest.sample(frac=0.5)
			test_data = rest.drop(development_data.index)
			
			for var in dependent_variables:			
				# balance the train set, and copy and balance the test set.
				train_data_balanced = Balance_Data(train_data,var)
				development_data_balanced = Balance_Data(development_data,var)
				development_data_unbalanced = development_data
				test_data_balanced = Balance_Data(test_data,var)
				test_data_unbalanced = test_data
				
				# split y from the rest of the data
				train_data_balanced_y = train_data_balanced[var]
				development_data_balanced_y = development_data_balanced[var]
				development_data_unbalanced_y = development_data_unbalanced[var]
				test_data_balanced_y = test_data_balanced[var]
				test_data_unbalanced_y = test_data_unbalanced[var]
				
				# loop through all feature group combinations
				for size in range(len(feature_groups)+1):
					for set in combinations(feature_groups,size):
						if set != ():
							
							development_data_balanced_X = pd.DataFrame()
							development_data_unbalanced_X = pd.DataFrame()
							train_data_balanced_X = pd.DataFrame()
							test_data_balanced_X = pd.DataFrame()
							test_data_unbalanced_X = pd.DataFrame()
							for tuple in set:
								for feature in tuple:
									# make X
									development_data_balanced_X[feature] = development_data_balanced[feature]
									development_data_unbalanced_X[feature] = development_data_unbalanced[feature]
									train_data_balanced_X[feature] = train_data_balanced[feature]
									test_data_balanced_X[feature] = test_data_balanced[feature]
									test_data_unbalanced_X[feature] = test_data_unbalanced[feature]
				
							# loop through all parameter settings and find the best option for this fold
							parameter_results_balanced = dict()
							parameter_results_unbalanced = dict()
							for parameter_settings in [(0.05,4),(0.05,6),(0.05,8),(0.1,4),(0.1,6),(0.1,8),(0.2,4),(0.2,6),(0.2,8)]:
								eta = parameter_settings[0]
								depth = parameter_settings[1]
					
								# parameter settings
								classifier = xgb.XGBClassifier(learning_rate = eta, max_depth = depth,
																n_estimators = 800 , n_jobs = -1, 
																objective = 'binary:logistic', silent = True)
								# train model on train set
								classifier.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")
								# test on development set
								y_proba_balanced = classifier.predict_proba(development_data_balanced_X)[:,1]
								y_proba_unbalanced = classifier.predict_proba(development_data_unbalanced_X)[:,1]

								auc_balanced = roc_auc_score(development_data_balanced_y,y_proba_balanced)
								auc_unbalanced = roc_auc_score(development_data_unbalanced_y,y_proba_unbalanced)

								parameter_results_balanced[parameter_settings]=auc_balanced
								parameter_results_unbalanced[parameter_settings]=auc_unbalanced
							
							# train a model on the train set with the optimal parameter settings and test it on the test set
							best_parameter_settings_balanced = keywithmaxval(parameter_results_balanced)
							best_parameter_settings_unbalanced = keywithmaxval(parameter_results_unbalanced)

							best_classifier_balanced = xgb.XGBClassifier(learning_rate = best_parameter_settings_balanced[0], 
																max_depth = best_parameter_settings_balanced[1],
																n_estimators = 800 , n_jobs = -1, 
																objective = 'binary:logistic', silent = True)
							best_classifier_unbalanced = xgb.XGBClassifier(learning_rate = best_parameter_settings_unbalanced[0], 
																max_depth = best_parameter_settings_unbalanced[1],
																n_estimators = 800 , n_jobs = -1, 
																objective = 'binary:logistic', silent = True)
							
							best_classifier_balanced.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")
							best_classifier_unbalanced.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")

							#calculate AUC for balanced test set
							y_proba_balanced = best_classifier_balanced.predict_proba(test_data_balanced_X)[:,1]
							y_val_balanced = [1. if y_cont > .5 else 0. for y_cont in y_proba_balanced]
							auc_balanced = roc_auc_score(test_data_balanced_y,y_proba_balanced)
							f1_balanced = f1_score(test_data_balanced_y,y_val_balanced)
							variable_combinations_results_balanced[(var,set)].append((auc_balanced, best_parameter_settings_balanced, f1_balanced))
							interim.write("B;%s;%s;%s;%s;%s;%s\n" %(str(var),str(set),str(auc_balanced),str(best_parameter_settings_balanced[0]),str(best_parameter_settings_balanced[1]),str(f1_balanced)))
							
							# calculate AUC for unbalanced test set
							y_proba_unbalanced = best_classifier_unbalanced.predict_proba(test_data_unbalanced_X)[:,1]
							y_val_unbalanced = [1. if y_cont > .5 else 0. for y_cont in y_proba_unbalanced]
							auc_unbalanced = roc_auc_score(test_data_unbalanced_y,y_proba_unbalanced)
							f1_unbalanced = f1_score(test_data_unbalanced_y,y_val_unbalanced)
							variable_combinations_results_unbalanced[(var,set)].append((auc_unbalanced, best_parameter_settings_unbalanced, f1_unbalanced))
							interim.write("B;%s;%s;%s;%s;%s;%s\n" %(str(var),str(set),str(auc_unbalanced),str(best_parameter_settings_unbalanced[0]),str(best_parameter_settings_unbalanced[1]),str(f1_unbalanced)))
			
			# write to file in-between
			with open(os.path.join(path_out,'variable_combinations_results_balanced8.p'), 'w') as file:
				file.write(pickle.dumps(variable_combinations_results_balanced))
			with open(os.path.join(path_out,'variable_combinations_results_balanced8.txt'), 'w') as file:
				file.write(str(variable_combinations_results_balanced))
			
			with open(os.path.join(path_out,'variable_combinations_results_unbalanced8.p'), 'w') as file:
				file.write(pickle.dumps(variable_combinations_results_unbalanced))
			with open(os.path.join(path_out,'variable_combinations_results_unbalanced8.txt'), 'w') as file:
				file.write(str(variable_combinations_results_unbalanced))
			# en nu naar de volgende fold
		# en wanneer je alle folds gehad hebt:
		else:
			print "that's 8!"
	 
#results = pickle.load( open( "variable_combinations_results_unbalanced.txt", "r" ) )
#results = json.load( open( "file.txt", "r" ) )