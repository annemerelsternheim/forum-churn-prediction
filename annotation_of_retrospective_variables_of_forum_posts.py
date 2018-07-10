# coding: utf-8

# This file was used to annotate the retrospective features of forum posts.
# It assumes that the static features were already annotated with "annotation_of_static_variables_in_forum_posts.py"

############################################################
## IMPORTS
############################################################
import glob, os, re, sys, json
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

############################################################
## FUNCITON DEFINITIONS
############################################################
def sum_of_earlier_inactivities(x,y):
	""" this function sums the inactivity variable over the past 30 days, recursively """
    if y == 1:
        return  x
    else:
        return x+sum_of_earlier_inactivities(x-1,y-1) if x>1 else 1

def make_first_access_dict(first_access_dict = dict()):
    for u in users:
        first_access_dict[u['user_id']] = u['created_date']
    return first_access_dict

def make_last_access_dict(last_access_dict = dict()):
    for u in users:
        last_access_dict[u['user_id']] = u['last_access']
    return last_access_dict
	
############################################################
## MAIN CODE
############################################################
path_in = r' ... \user-csvs'
path_out = r'... \user-csvs_retro'

users = json.load(open('D:\\4. Data\\Amazones\\nieuw_users_status.json'))

first_access_dict = make_first_access_dict()
last_access_dict = make_last_access_dict()
	
for filename in glob.glob(os.path.join(path_in, '*.csv')):
	user = re.search("_(.[0-9]*)\.csv", filename).group(1)
	first_access = datetime.strptime(first_access_dict[user], '%d/%m/%Y - %H:%M')
	last_access = datetime.strptime(last_access_dict[user], '%d/%m/%Y - %H:%M')
	
	# all new_... lists will eventually contain all retrospective feature values
	new_inactivity=[]
	new_sentences = []
	new_words = []
	new_questions = []
	new_sentiment=[]
	new_subjectivity=[]
	churn3 = [] # churn in three months
	churn2 = [] # churn in two months
	churn1 = [] # churn in one month
	
	df = pd.read_csv(filename)
	
	dates = df['Date & Time']
	inactivity = df['Inactivity']
	sentences = df['Sentences/Post'] # post length (in sentences)
	words = df['Words/Sentence'] # sentence length (in words)
	questions = df['Questions']
	sentiment = df['Sentiment']
	subjectivity = df['Subjectivity']
	
	# keep track of when last post was. initialisation
	last_active_bin = first_access
	
	first = min([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates])
	last = max([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates])
	binlist = [first+(bin*timedelta(days=1)) for bin in range((last-first).days)] 
	earlier = [bin+relativedelta(months=-1) for bin in binlist]
	
	
	for bin in binlist:
		churn3.append(1 if bin+relativedelta(months=3)>last_access else 0)
		churn2.append(1 if bin+relativedelta(months=2)>last_access else 0)
		churn1.append(1 if bin+relativedelta(months=1)>last_access else 0)
		
		# Turn the date strings in the date column into datetime objects for easy comparisons
		datesDT = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
		
		if bin in datesDT:
			# fill all filled bins with their original value
			new_sentences.append(sentences[datesDT.index(bin)])
			new_words.append(words[datesDT.index(bin)])
			new_questions.append(questions[datesDT.index(bin)])
			new_sentiment.append(sentiment[datesDT.index(bin)])
			new_subjectivity.append(subjectivity[datesDT.index(bin)])
			if last_active_bin == first_access:
				new_inactivity.append((sum_of_earlier_inactivities(inactivity[datesDT.index(bin)],
										(binlist[1]-earlier[1]).days))/(binlist[1]-earlier[1]).days)
			else:
				new_inactivity.append(inactivity[datesDT.index(bin)])
			last_active_bin = bin
		else:
			# fill empty bins with value 0, except inactivity:
			new_inactivity.append((bin-last_active_bin).days-1)
			new_sentences.append(0)
			new_words.append(0)
			new_questions.append(0)
			new_sentiment.append(0)
			new_subjectivity.append(0)
			
	#concatenate these variables into a dataframe (near identical to the ones made with file "annotation_of_static_variables_in_forum_posts.py", but including empty time bins)
	df = pd.DataFrame({ "Date & Time": binlist,
						"Churn in 3m": churn3 ,"Churn in 2m": churn2 ,"Churn in 1m": churn1 ,
						"Inactivity": new_inactivity,"Sentences": new_sentences, "Questions": new_questions,
						"Sentiment": new_sentiment, "Subjectivity":new_subjectivity, "Words": new_words })
	
	# this is the relevant part: determine the average values over the past month for these features.
	binlist = df['Date & Time']
	
	inactivity = df['Inactivity']
	sentences = df['Sentences']
	words = df['Words']
	questions = df['Questions']
	sentiment = df['Sentiment']
	subjectivity = df['Subjectivity']
	
	first = min([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for bin in binlist])
	last = max([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for bin in binlist])
	earlier = [bin+relativedelta(months=-1) for bin in binlist]
	
	# the columns with averaged values that will end up in the new csv files
	Inactivity = []
	Sentences = []
	Words = []
	Questions = []
	Sentiment = []
	Subjectivity = []

	# loop nu door de hele lijst. voeg aan bin x de waarde van x-1 tm x-29 toe
	for i in range(len(enumerate(binlist)):
		# the 30 values of the last month
		inactivity30 = []
		sentences30 = []
		words30 = []
		questions30 = []
		sentiment30 = []
		subjectivity30 = []
		
		for j in range((binlist[i]-earlier[i]).days):
			if i-j >= 0:
				inactivity30.append(inactivity[i-j])
				sentences30.append(sentences[i-j])
				words30.append(words[i-j])
				questions30.append(questions[i-j])
				sentiment30.append(sentiment[i-j])
				subjectivity30.append(subjectivity[i-j])

		Inactivity.append(np.mean(inactivity30))
		Sentences.append(np.mean(sentences30))
		Words.append(np.mean(words30))
		Questions.append(np.mean(questions30))
		Sentiment.append(np.mean(sentiment30))
		Subjectivity.append(np.mean(subjectivity30))
	
	# append new columns to existing columns
	df['Sentence mean past 30 days'] = Sentences
	df['Word mean past 30 days'] = Words
	df['Inactive mean past 30 days'] = Inactivity
	df['Questions mean past 30 days'] = Questions
	df['Sentiment mean past 30 days'] = Sentiment
	df['Subjectivity mean past 30 days'] = Subjectivity
	
	# name dataframe and write to file
	name = str(user)+"_churn123.csv"
	df.to_csv(os.path.join(path_out,name),index=False)