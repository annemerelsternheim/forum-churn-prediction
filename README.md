# forum-churn-prediction
This repo contains the code that I wrote for the experiments that I did during my master's thesis. I predicted churn in a social community with easy-to-annotate variables that were expected to have predictive value, using the supervised learning algorithm XGBoost.

The goal of the thesis was to find out which variables would predict churn of patients (one, two and three months into the future) on a forum best. Annotated variables were: Inactivity (time since last post), Sentiment, Subjectivity, Questions (as ratio of sentences), Sentence length and Post length. These features were also summarised for the past month: retrospective variables. The dependent and independent variables were combined in 93 unique ways, which resulted in 93 different models per experiment. Performance was evaluated wit AUC and FNR.

This repository contains the following files:

	1) posts.json
contains an example forum post of one example user, copied several times as to imitate frequent posting activity

	2) users.json
contains two example users: one that has posted, one that has not

	3) annotation_of_static_variables_in_forum_posts.py
contains the code with which the six static features were annotated. Input: posts.json and users.json. Output: one csv file with annotations per user. In separate folder.

	4) annotation_of_retrospective_variables_of_forum_posts.py
contains the code with which the retrospective features were annotated. Input: users.json and the folder with csv files that was the result from 3). Output: one csv file with more annotations per user. In separate folder.
	
	5) choosing_parameters_for_gridsearch.py
contains the code that was used to choose the parameter options that had to be passed in the experiment. Uses 5-fold cross-validation.
	
	6) XGBoost_on_all_variable_combinations.py
contains the code that generates the 93 different variable combinations, and trains and tests XGBoost models on these. Input: the csv files resulting from 5). Output: a dataframe with all results.

More code was used to visualise the results from the dataframe. I plan to add this code later.
