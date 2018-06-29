	
	# permuteren
	# per split train test ongebalanceerd
	# wegen op train test op test


# split data into training and test set
# use training set for parameter optimisation, test set for validation etc
# on training set: 	select some features (according to a predefined loop),
#							select one of three dependent variables (churn in 1 2 or 3 months)
#								determine optimal parameters for this combination with 10-fold cross validation
# using 10-fold cross validation:
#	fit the best_estimator model on a training part of the test set,
#	and test it on the remaining test part of the test set.
# save the AUC value for every fold in a list, and save it as part of a giganteous data frame 

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
import warnings # in order to suppress the deprecation warning raised by stratifiedKFold
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from tqdm import *

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
	return data
	
def Balance_Data(data, dependent):
	""" function needs a Pandas data frame, selects only the positive values from the dependent variable, and selects as many negatives.
	Returns the concatenated data frame of these """
	print "Balancing data..."
	positives = data[(data[dependent]==1)]
	negatives = data.drop(positives.index).sample(n=len(positives))
	balanced_data = pd.concat([positives,negatives])
	return balanced_data
	
def Shuffle_Data(data,dependent):
	print "Shuffling data..."
	shuffled_data = data.sample(frac=1)
	return shuffled_data
	
	
def Do_grid_search(train_X,train_y):
	""" function needs the X and y parts of the training data."""
	print "Searching for the optimal set of parameters... (can take several minutes!)"
	neg = float(sum([1 for y in train_y if y==0]))
	print neg
	pos = float(sum([1 for y in train_y if y==1]))
	print pos
	parameters = {'objective':['binary:logistic'],
				  'learning_rate': [0.05,0.1,0.2], #so called `eta` value
				  'max_depth': [4,6,8], 'silent': [True],
				  'n_estimators': [800],
				  'scale_pos_weight': [neg/pos]}
				  
	print "the ratio negative to positive samples is %.4f" %(neg/pos)
			  
	clf = GridSearchCV(xgb.XGBClassifier(), #the estimator object
					   parameters, #the parameter grid
					   cv=StratifiedKFold(n_splits=10, shuffle=True).split(train_X,train_y), 
					   scoring='roc_auc')
	clf.fit(train_X, train_y)
	return clf

def Split_data(train_index,test_index, X,y):
	""" function needs the train and test indexes (together with the to-be-splitted X and y), and will return test and training sets of both X and y."""
	print "Separating X from y..."
	X = X.reset_index(drop=True)
	y = y.reset_index(drop=True)
	
	train_X = X.drop(test_index)
	test_X = X.drop(train_index)
	train_y = y.drop(test_index)
	test_y = y.drop(train_index)
	return(train_X,test_X,train_y,test_y)

######################################################################
# MAIN
######################################################################

path_in = "C:\Users\sternheimam\Desktop\my-notebook\user-csvs_predictions123"
path_out = "C:\Users\sternheimam\Desktop\my-notebook"

#feature_names = ["inactivity","questions","sentences","sentiment","subjectivity","words",
#                 "sentence mean","word mean", "inactive mean","questions mean","sentiment mean","subjectivity mean"]
feature_groups = [(3,11),(13,14),(6,7),(9,10,12),(4,5,8)] # inactivity, opinionmining past, opinionmining, textual past, textual
dependent_variables = [0,1,2]

# get the data from the files
data = Get_data(path_in) # all files
# initialise the matrix/dataframe that will eventually contain all data
The_Matrix = pd.DataFrame()

#visualise progress. Total number of possible combinations in 93. Add 1 to progress bar after testing of a feature combination
with tqdm(total=93) as pbar:
	# for all dependent variables (0,1,2: 1 2 or 3 months churn)
	for var in dependent_variables:
		column_values = []
		# balance the data
		#balanced_data = Balance_Data(data,var) # all positive samples, and equally many negative samples
		balanced_data = Shuffle_Data(data,var) # all samples, in random order
		# the dependent variable is the column from the data which has that dependent var name (0,1 or 2)
		dependent_var = balanced_data[var]
		
		# generate all possible combinations of independent-var groups
		for size in range(len(feature_groups)+1):
			for combi in combinations(feature_groups,size): # combinations is a list of lists with tuples from feature names in it
				# ignore the empty set: you can not predict the dependent variable without independent variables.
				if combi != ():
					# initialise the new data frame (if there are independent variables)
					frame = pd.DataFrame()
					# add columns to it: you had already saved the dependent variable, the others are extracted from the training data
					for tup in combi:
						for feature in tup:
							frame[feature] = balanced_data[feature]
					# the dependent and independent variables of the data frame are now collected
					# rename and reindex them
					X = frame.reset_index(drop=True)
					y = dependent_var.reset_index(drop=True)
				
					# split the data into a training ans testing set: the training set is used for grid search, the test set for the rest
					trainparams_X,rest_X,trainparams_y,rest_y = train_test_split(X,y)
					# determine the optimal parameters with grid search
					classifier = Do_grid_search(trainparams_X,trainparams_y)
					
					
					# the best classifier is now determined. Namely:
					best_classifier = classifier.best_estimator_
					# save the parameter settings, and intitialise a list to save auc per fold
					optimal_params = best_classifier.get_params()
					all_auc = []
					# split the rest-data in ten folds
					ss = ShuffleSplit(n_splits=10)
					for train_index, test_index in ss.split(rest_X):
					# split training from testing set
						train_X,test_X,train_y,test_y = Split_data(train_index,test_index,rest_X,rest_y)
						# train the best classifier (determined by grid search) on the training part of the rest-data
						best_classifier.fit(train_X,train_y,eval_metric="auc")
						# and test its predictive accuracy on the test part of the rest-data
						y_probabilities = best_classifier.predict_proba(test_X)[:,1] # array containing the probability that dependent_var==1
						auc = roc_auc_score(test_y,y_probabilities)
						# append the score to a list that will contain all auc scores for this particular 10-fold-run
						all_auc.append(auc)
					# 10-fold has now finished. Append all_auc to a list containing all scores for the current dependent variable
					column_values.append({"parameter_settings": optimal_params, "AUC_scores": all_auc})
					# add 1 to progress bar
					pbar.update(1)
		# name the column, and add it to the super dataframe / matrix
		column_name = "Churn in %s months" %(int(var)+1)
		The_Matrix[column_name]=column_values
	The_Matrix.to_csv(os.path.join(path_out,"testMATRIX.csv"),index=False,sep=";")









