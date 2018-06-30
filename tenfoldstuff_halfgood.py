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
from sklearn.metrics import roc_auc_score # for determining the predictive accuracy of the model
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
	
def keywithmaxval(dictionary):
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

feature_groups = [(3,11),(13,14),(6,7),(9,10,12),(4,5,8)] # inactivity, opinionmining past, opinionmining, textual past, textual
dependent_variables = [0,1,2]

# get the data from the files
data = Get_data(path_in) # all files

variable_combinations_results_10fold = defaultdict(list)

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
			# split y from the rest of the data
			development_data_y = development_data[var]
			train_data_y = train_data[var]
			test_data_y = test_data[var]
			
			neg = float(sum([1 for y in train_data_y if y==0]))
			pos = float(sum([1 for y in train_data_y if y==1]))
			
			# loop through all feature group combinations
			for size in range(len(feature_groups)+1):
				for set in combinations(feature_groups,size):
					if set != ():
						
						development_data_X = pd.DataFrame()
						train_data_X = pd.DataFrame()
						test_data_X = pd.DataFrame()
						for tuple in set:
							for feature in tuple:
								# make X
								development_data_X[feature] = development_data[feature]
								train_data_X[feature] = train_data[feature]
								test_data_X[feature] = test_data[feature]
			
						# loop through all parameter settings and find the best option for this fold
						parameter_results = dict()
						for parameter_settings in [(0.05,4),(0.05,6),(0.05,8),(0.1,4),(0.1,6),(0.1,8),(0.2,4),(0.2,6),(0.2,8)]:
							eta = parameter_settings[0]
							depth = parameter_settings[1]
							
							# parameter settings
							classifier = xgb.XGBClassifier(learning_rate = eta, max_depth = depth,
															n_estimators = 800 , n_jobs = -1, 
															objective = 'binary:logistic', silent = True,
															scale_pos_weight = neg/pos)
							# train model on train set
							classifier.fit(train_data_X,train_data_y,eval_metric="auc")
							# test on development set
							y_proba = classifier.predict_proba(development_data_X)[:,1]
							auc = roc_auc_score(development_data_y,y_proba)
							parameter_results[parameter_settings]=auc
						
						# train a model on the train set with the optimal parameter settings and test it on the test set
						best_parameter_settings = keywithmaxval(parameter_results)
						best_classifier = xgb.XGBClassifier(learning_rate = best_parameter_settings[0], 
															max_depth = best_parameter_settings[1],
															n_estimators = 800 , n_jobs = -1, 
															objective = 'binary:logistic', silent = True,
															scale_pos_weight = neg/pos)
						best_classifier.fit(train_data_X,train_data_y,eval_metric="auc")
						
						# calculate AUC on test set
						y_proba = classifier.predict_proba(test_data_X)[:,1]
						auc = roc_auc_score(test_data_y,y_proba)
						variable_combinations_results_10fold[(var,set)].append((auc, best_parameter_settings))
		
		# write to file in-between
		with open(os.path.join(path_out,'variable_combinations_results_10fold.p'), 'w') as file:
			file.write(pickle.dumps(variable_combinations_results_10fold))		
		with open(os.path.join(path_out,'variable_combinations_results_10fold.txt'), 'w') as file:
			file.write(str(variable_combinations_results_10fold))
			
		print "RESULTS: ", variable_combinations_results_10fold
		print
		# en nu naar de volgende fold
	# en wanneer je alle folds gehad hebt:
	else:
		print "that's it!"

# now write the whole #! to a file. Pickle serialises the dictionary.
		with open(os.path.join(path_out,'variable_combinations_results_10fold.p'), 'w') as file:
			file.write(pickle.dumps(variable_combinations_results_10fold))		
		with open(os.path.join(path_out,'variable_combinations_results_10fold.txt'), 'w') as file:
			file.write(str(variable_combinations_results_10fold))

	 
#results = pickle.load( open( "variable_combinations_results_10fold.txt", "r" ) )
#results = json.load( open( "file.txt", "r" ) )