from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import xgboost as xgb
import pandas as pd
import os
from sklearn import preprocessing # for minmaxscaler


def Get_data(path,files=[]):
	""" function needs a path to a folder where all csv-files-to-be-concatenated are placed, and returns a Pandas data frame """
	print "Getting data..."
	for filename in os.listdir(path):
		files.append(pd.read_csv(os.path.join(path, filename)))
	data=pd.concat(files)
	data = data.drop('Date & Time',axis=1)
	data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))
	return data

if __name__ == "__main__":
    
	path_in = "C:\Users\sternheimam\Desktop\my-notebook\user-csvs_predictions123"
	data = Get_data(path_in) # all files

	data_small = data.sample(frac=0.005)
	data_small_X = data_small.drop([1,2,3],axis=1).as_matrix()
	data_small_y = data_small[2].as_matrix()
		
	# Number of random trials
	NUM_TRIALS = 3

	# Load the dataset
	X_iris = data_small_X
	y_iris = data_small_y

	# Set up possible values of parameters to optimize over
	p_grid = {'objective':['binary:logistic'],
					  'learning_rate': [0.05,0.1],#,0.2], #so called `eta` value
					  'max_depth': [4],#,6,8], 'silent': [True],
					  'n_estimators': [800]}

	# We will use a Support Vector Classifier with "rbf" kernel
	estim = xgb.XGBClassifier()
	#SVC(kernel="rbf")

	# Arrays to store scores
	non_nested_scores = np.zeros(NUM_TRIALS)
	nested_scores = np.zeros(NUM_TRIALS)

	# Loop for each trial
	for i in range(NUM_TRIALS):

		# Choose cross-validation techniques for the inner and outer loops,
		# independently of the dataset.
		# E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
		inner_cv = KFold(n_splits=2, shuffle=True, random_state=i)
		outer_cv = KFold(n_splits=2, shuffle=True, random_state=i)

		# Non_nested parameter search and scoring
		print "Doing non-nested"
		clf = GridSearchCV(estimator=estim, param_grid=p_grid, cv=inner_cv,scoring = "roc_auc", n_jobs = -1)
		clf.fit(X_iris, y_iris)
		non_nested_scores[i] = clf.best_score_

		print "Doing nested"
		# Nested CV with parameter optimization
		nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv, scoring = "roc_auc")
		nested_scores[i] = nested_score.mean()

	print non_nested_scores
	print nested_scores
	score_difference = non_nested_scores - nested_scores




	print("Average difference of {0:6f} with std. dev. of {1:6f}."
		  .format(score_difference.mean(), score_difference.std()))

	# Plot scores on each trial for nested and non-nested CV
	plt.figure()
	plt.subplot(211)
	non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
	nested_line, = plt.plot(nested_scores, color='b')
	plt.ylabel("score", fontsize="14")
	plt.legend([non_nested_scores_line, nested_line],
			   ["Non-Nested CV", "Nested CV"],
				bbox_to_anchor=(0, .4, .5, 0))
	plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
			  x=.5, y=1.1, fontsize="15")

	# Plot bar chart of the difference.
	plt.subplot(212)
	difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
	plt.xlabel("Individual Trial #")
	plt.legend([difference_plot],
			   ["Non-Nested CV - Nested CV Score"],
				bbox_to_anchor=(0, 1, .8, 0))
	plt.ylabel("score difference", fontsize="14")

	plt.show()
