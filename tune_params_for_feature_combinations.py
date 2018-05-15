# split data into training and test set
# use training set for parameter optimisation, test set for validation etc
# on training set: 	select some features (according to a predefined loop),
#					select one of three dependent variables (churn in 1 2 or 3 months)
#					determine optimal parameters for this combination with 10-fold cross validation
# using 10-fold cross validation:
#	fit the best_estimator model on a training part of the test set,
#	and test it on the remaining test part of the test set.
# save the AUC value for every fold in a list, and save it as part of a giganteous data frame 

######################################################################
# IMPORTS
######################################################################
import os # for path
import pandas as pd # for dataframes
from sklearn import preprocessing # for minmaxscaler

######################################################################
# FUNCTION DEFINITIONS
######################################################################

def Get_data(path,files=[]):
	print "Getting data..."
	for filename in os.listdir(path):
		files.append(pd.read_csv(os.path.join(path, filename)))
	data=pd.concat(files)
	data = data.drop('Date & Time',axis=1)
	data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))
	return data

def Split_into_Train_and_Test(data):
	print "Splitting data..."
	pass

######################################################################
# MAIN
######################################################################

path = "C:\Users\sternheimam\Desktop\my-notebook\user-csvs_predictions"
print len(Get_data(path))