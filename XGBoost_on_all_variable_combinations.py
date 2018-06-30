######################################################################
# DESCRIPTION
######################################################################

# In this code, XGBoost is applied to different combinations of dependent and independent variables.
# It was the crux of my thesis, as this performed all my experiments.
# the output is further analysed through visualisation

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
from sklearn.metrics import roc_auc_score, confusion_matrix # for determining the predictive accuracy of the model
from collections import defaultdict
import cPickle as pickle
import json
from datetime import datetime


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
	
def keywithmaxval(dictionary): # from https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary/12343826#12343826
	""" a) create a list of the dict's keys and values; 
	 b) return the key with the max value"""  
	v=list(dictionary.values())
	k=list(dictionary.keys())
	return k[v.index(max(v))]
	
######################################################################
# MAIN
######################################################################

path_in = "./csvs"
path_out = "./output"

# different combinations of these feature groups are used to predict churn with.
feature_groups = [(3,11),(13,14),(6,7),(9,10,12),(4,5,8)]
dependent_variables = [0,1,2] # churn in 1, 2, 3 months

# get the data from the files
data = Get_data(path_in)

variable_combinations_results_balanced = defaultdict(list)
variable_combinations_results_unbalanced = defaultdict(list)

folds = 10
bounds = [int(s*(len(data)/folds)) for s in range(folds+1)] # if len(data) is not congruent modulo 'folds' (not strictly devidable by the nr of folds), then the tail (modulo) is simply ignored.

with open(os.path.join(path_out,'COPY_FIRST_if_code_is_running.txt'),"w") as interim:
	interim.write("B/U;dependent;independent;AUC;eta;depth;cm\n") # I like to write some results to a file in-between. This way I can check whether the program is actually doing something.

	for i,b in enumerate(bounds):
		if bounds[i]!=bounds[-1]:
			start_fold = datetime.now()
			print "FOLD %i started at %s" %(i,str(start_fold))
			
			fold_data = data[bounds[i]:bounds[i+1]]

			# split the current fold into three parts: train, development, and test set
			train_data = fold_data.sample(frac=0.6)
			rest = fold_data.drop(train_data.index)
			development_data = rest.sample(frac=0.5)
			test_data = rest.drop(development_data.index)
			
			for var in dependent_variables: # churn is predicted one, two and three months into the future	
				# balance the train set, and copy and balance the development and test set.
				train_data_balanced = Balance_Data(train_data,var)
				development_data_balanced = Balance_Data(development_data,var)
				development_data_unbalanced = development_data
				test_data_balanced = Balance_Data(test_data,var)
				test_data_unbalanced = test_data
				
				# split y (the dependent variable) from the rest of the data
				train_data_balanced_y = train_data_balanced[var]
				development_data_balanced_y = development_data_balanced[var]
				development_data_unbalanced_y = development_data_unbalanced[var]
				test_data_balanced_y = test_data_balanced[var]
				test_data_unbalanced_y = test_data_unbalanced[var]
				
				# loop through all feature group combinations
				for size in range(len(feature_groups)+1):
					for set in combinations(feature_groups,size):
						if set != (): # () corresponds to the empty set of feature groups
							development_data_balanced_X = pd.DataFrame()
							development_data_unbalanced_X = pd.DataFrame()
							train_data_balanced_X = pd.DataFrame()
							test_data_balanced_X = pd.DataFrame()
							test_data_unbalanced_X = pd.DataFrame()
							
							for tuple in set: # every tuple is a feature group
								for feature in tuple: # every feature in every group needs to be looked up in 'data', and added to X
									development_data_balanced_X[feature] = development_data_balanced[feature]
									development_data_unbalanced_X[feature] = development_data_unbalanced[feature]
									train_data_balanced_X[feature] = train_data_balanced[feature]
									test_data_balanced_X[feature] = test_data_balanced[feature]
									test_data_unbalanced_X[feature] = test_data_unbalanced[feature]
				
							parameter_results_balanced = dict()
							parameter_results_unbalanced = dict()
							# loop through all parameter settings and find the best option for this fold, with this dependent variable, and these feature groups.
							for (eta,depth) in [(eta,depth) for eta in [0.05,0.1,0.2]
															for depth in [8,10,12]]: # these parameter options were chosen with an initial exploration (choosing_parameters_for_gridsearch.py)
								# set parameters
								classifier = xgb.XGBClassifier(learning_rate = eta, max_depth = depth,
																n_estimators = 800 , n_jobs = -1, # I had 12 cores available. Run time was about an hour (using a balanced train set).
																objective = 'binary:logistic', silent = True)
								# train model on train set
								classifier.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")
								# test on development set
								y_proba_balanced = classifier.predict_proba(development_data_balanced_X)[:,1]
								y_proba_unbalanced = classifier.predict_proba(development_data_unbalanced_X)[:,1]

								auc_balanced = roc_auc_score(development_data_balanced_y,y_proba_balanced)
								auc_unbalanced = roc_auc_score(development_data_unbalanced_y,y_proba_unbalanced)

								parameter_results_balanced[(eta,depth)]=auc_balanced
								parameter_results_unbalanced[(eta,depth)]=auc_unbalanced
							
							# when all parameter combinations are tried, choose the settings that performed best...
							best_parameter_settings_balanced = keywithmaxval(parameter_results_balanced)
							best_parameter_settings_unbalanced = keywithmaxval(parameter_results_unbalanced)

							# ... and use these parameters
							best_classifier_balanced = xgb.XGBClassifier(learning_rate = best_parameter_settings_balanced[0], 
																max_depth = best_parameter_settings_balanced[1],
																n_estimators = 800 , n_jobs = -1, 
																objective = 'binary:logistic', silent = True)
							best_classifier_unbalanced = xgb.XGBClassifier(learning_rate = best_parameter_settings_unbalanced[0], 
																max_depth = best_parameter_settings_unbalanced[1],
																n_estimators = 800 , n_jobs = -1, 
																objective = 'binary:logistic', silent = True)
	
							# Train both classifiers on balanced data
							best_classifier_balanced.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")
							best_classifier_unbalanced.fit(train_data_balanced_X,train_data_balanced_y,eval_metric="auc")

							# Test one on a balanced test set...
							y_proba_balanced = best_classifier_balanced.predict_proba(test_data_balanced_X)[:,1] # probability of sample begin positive (0.0 - 1.0)
							y_val_balanced = [1. if y_cont > .5 else 0. for y_cont in y_proba_balanced] # label given to samples: positive (1) or negative (0)
							auc_balanced = roc_auc_score(test_data_balanced_y,y_proba_balanced)
							cm_b = confusion_matrix(test_data_balanced_y,y_val_balanced).ravel() #tn, fp, fn, tp
							variable_combinations_results_balanced[(var,set)].append((auc_balanced, best_parameter_settings_balanced, cm_b))
							interim.write("B;%s;%s;%s;%s;%s;%s\n" %(str(var),str(set),str(auc_balanced),str(best_parameter_settings_balanced[0]),str(best_parameter_settings_balanced[1]),str(cm_b)))
							
							# ...and the other on an unbalanced test set
							y_proba_unbalanced = best_classifier_unbalanced.predict_proba(test_data_unbalanced_X)[:,1]
							y_val_unbalanced = [1. if y_cont > .5 else 0. for y_cont in y_proba_unbalanced]
							auc_unbalanced = roc_auc_score(test_data_unbalanced_y,y_proba_unbalanced)
							cm_u = confusion_matrix(test_data_unbalanced_y,y_val_unbalanced).ravel() #tn, fp, fn, tp
							variable_combinations_results_unbalanced[(var,set)].append((auc_unbalanced, best_parameter_settings_unbalanced, cm_u))
							interim.write("B;%s;%s;%s;%s;%s;%s\n" %(str(var),str(set),str(auc_unbalanced),str(best_parameter_settings_unbalanced[0]),str(best_parameter_settings_unbalanced[1]),str(cm_u)))
			
			# write to file (I used the .p files to do further analyses on, but the .txt files are more human-readable)
			with open(os.path.join(path_out,'variable_combinations_results_balanced.p'), 'w') as file:
				file.write(pickle.dumps(variable_combinations_results_balanced)) # to read back: pickle.load(open("file.txt","r"))
			with open(os.path.join(path_out,'variable_combinations_results_balanced.txt'), 'w') as file:
				file.write(str(variable_combinations_results_balanced)) # to read back: json.load(open("file.txt","r"))
			
			with open(os.path.join(path_out,'variable_combinations_results_unbalanced.p'), 'w') as file:
				file.write(pickle.dumps(variable_combinations_results_unbalanced))
			with open(os.path.join(path_out,'variable_combinations_results_unbalanced.txt'), 'w') as file:
				file.write(str(variable_combinations_results_unbalanced))

			stop_fold = datetime.now()
			print "\t finished at %s" %str(stop_fold)
			time_diff = stop_fold-start_fold
			print "\t took %i minutes \n" %(time_diff.total_seconds()/60)
			
		# all folds are done!
		else:
			print "That's it! \n"