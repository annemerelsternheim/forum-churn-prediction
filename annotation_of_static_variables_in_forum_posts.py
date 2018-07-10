# coding: utf-8

# This is the code that annotates the static features in the forum posts. Another file annotates the retrospective features
# static features that are annotated, are: inactivity, sentiment, objectivity, sentence length (in words), post length (in sentences) and questions (as ratio sentences)

############################################################
## IMPORTS
############################################################

import json, re, unidecode, os
from collections import defaultdict
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from pattern.nl import sentiment
import pandas as pd


############################################################
## FUNCTION DEFINITIONS
############################################################

#---------------------------
# Prepare text in .json files for analysis (probably more than needed, but might useful for future applications or analyses)
#---------------------------
def remove_non_ascii(text):
	""" this function expects a string, and removes non-ascii characters from it """
	return unidecode.unidecode(text)
	
def cleanup(text):
	""" this function expects a string (post from the BVN/Amazones forum), and returns a cleaner version of it """
	# remove all links, images, quotes, and emailaddresses
	text=re.sub('<a.*?>(.*?)</a>','',text) #remove links by HTML markup
	text=re.sub('(http:|www)\S*','',text) #remove links without markup
	text=re.sub('\[\\\/url\]','',text)
	text=re.sub('<img.*?/>', '',text) #remove images by HTML markup
	text=re.sub('<div class="bb-quote">((\s|\S)*?)</div>','',text) #remove quotes by HTML markup
	text=re.sub('<script.*?>([\S\s]*?)</script>','',text) #remove emailaddresses by HTML markup

	# replace all emoticon-icons by HTML markup
	text=re.sub('<img.*?title="(.*?)".*?/>', '(EMO:\\1)',text) #replace emoticons by textual indicators 

	# replace (most) sideways latin emoticons (https://en.wikipedia.org/wiki/List_of_emoticons shows an overview of smileys and their meanings)
	text=re.sub('[^>]:-?(\)|\])','(EMO:smiley)',text)
	text=re.sub(u'☺️','(EMO:smiley)',text)
	text=re.sub('[^>]:-?(\(|\[)','(EMO:sad)',text)
	text=re.sub(';-?(\)|\])','(EMO:wink)',text)
	text=re.sub(r'(:|;|x|X)-?(D)+\b','(EMO:laugh)',text)
	text=re.sub(':-?(/|\\\|\|)','(EMO:frown)',text)
	text=re.sub(r'(:|;)-?(p|P)+\b','(EMO:cheeky)',text)
	text=re.sub('(:|;)(\'|\")-?(\(|\[)','(EMO:cry)',text)
	text=re.sub('\<3+','(EMO:heart)',text)
	text=re.sub(u'❤️','(EMO:heart)',text)
	text=re.sub('((\>:-?(\(|\]))|(\>?:-?@))','(EMO:angry)',text)
	text=re.sub('\>:-?(\)|\])','(EMO:evil)',text)
	text=re.sub(r'(:|;)-?(O|o|0)+\b','(EMO:shock)',text)
	text=re.sub('(:|;)-?(K|k|x|X)','(EMO:kiss)',text)

	#other adjustments:
	text=re.sub('m\'?n\s','mijn ',text) # replacing m'n and mn with mijn, so it gets parsed correctly.
	text=re.sub('z\'?n\s','zijn ',text) #replacing z'n and zn with zijn
	text=re.sub('d\'?r\s','haar ',text) #replacing d'r and dr with zijn (only if followed by space, so dr. stays dr.)

	# replace all emoticons (and other things) written between double colons
	text=re.sub(':([a-zA-Z]+):','(EMO:\\1)',text)

	# remove remaining markup
	text=re.sub('</?(ol|style|b|p|em|u|i|strong|br|span|div|blockquote|li)(.*?)/?>','',text)
	text=re.sub('(\[|\]|\{|\})', '',text)

	# punctuation sometimes sticks to text where it shouldn't, so this code adds spaces in between text and punctuation
	text = re.sub('(\.{2,}|/|\)|,|!|\?)','\\1 ',text) # space behind
	text = re.sub('(/|\()',' \\1',text) # space in front
	text = re.sub('(\w{2,})(\.|,)','\\1 \\2 ',text) #space 'between'

	return(remove_non_ascii(text))

#---------------------------
# Go through data, save in dictionary
#---------------------------
def make_P_T_and_D(posts):
	""" this function takes the .json files containing the thread starts and responses, and returns three things:
	P: a dictionary with the user-ID as key, and the post as value;
	T: a dictionary with the user-ID as key, and the time of posting as a value;
	D: a list of all datetimes present in the data (sorted by date, because the .json was already sorted) """
	P = defaultdict(list)
	T = defaultdict(list)
	D = []

	print "Loading ..."
	for p in reversed(posts):
		P[p['user_id']].append(cleanup(p["body"]))
		T[p['user_id']].append(p['post_date'])
		D.append(datetime.strptime(p['post_date'], '%d/%m/%Y - %H:%M'))
	return (P,T,D)

#---------------------------
# Make list of bin-bondaries-to-be (binlist)
#---------------------------
def make_binlist(D,bin_hours=1):
	""" this function takes a list of dates (D), and generates a new list of dates,
	starting at 4:00 AM just before the earliest date in D, and ending at 4:00 just after the latest date in D,
	with fixed timeticks between all dates in the list.
	Optionally, the length of the timetick may be specified (in hours).
	"""
	#set lower and upper boundaries of a user's activity
	lower = min(D)
	upper = max(D)
	
	if lower.time()>=datetime.strptime('4:00','%H:%M').time():
		lower = lower.replace(hour = 4, minute = 0)
	else:
		lower = (lower+timedelta(days = -1)).replace(hour=4,minute=0)

	if upper.time()<datetime.strptime('12:00','%H:%M').time():
		upper = upper.replace(hour = 4, minute = 0)
	else:
		upper = (upper+timedelta(days=1)).replace(hour=4,minute=0)

	return([lower + timedelta(hours=x) for x in range(0, 24*((upper-lower).days), hours)])
	
#---------------------------
# Determine first activity of all users
#---------------------------
def make_first_access_dict(first_access_dict = dict()):
	for u in users:
		first_access_dict[u['user_id']] = u['created_date']
	return first_access_dict

#---------------------------
# Determine last log-on of all users
#---------------------------
def make_last_access_dict(last_access_dict = dict()):
	for u in users:
		last_access_dict[u['user_id']] = u['last_access']
	return last_access_dict

#---------------------------
# User selection
#---------------------------
def determine_active_users(include = [u['user_id'] for u in users], exclude = []):
	""" The users included in the experiment are assembled here. Omit users with less than 30 posts, or others you have explicitly specified. """
	return [user for user in include if len(T[user])>=30 and user not in exclude]

#---------------------------
# Variable annotation
#---------------------------
def determine_questionmarks(body, Q=0):
	""" This function counts and returns the number of sentences in the provided input string
	that ends in at least one question mark """
	for sentence in sent_tokenize(body):
		if re.search('\?+', sentence):
			Q+=1
	return float(Q)/float(len(sent_tokenize(body))) if len(sent_tokenize(body))!=0 else 0

def determine_sentiment(body):
	""" this funciton determines and returns the average sentiment of sentences in the provided input string.
	It uses the pattern module to do so. Sentiment values may range from -1 to 1. """
	return np.mean([sentiment(sentence)[0] for sentence in sent_tokenize(body)]) 

def determine_subjectivity(body):
	""" This function determines and returns the average subjectivity of sentences in the provided input string.
	It uses the pattern module to do so. Subjectivity values may range from 0 to 1. """
	return np.mean([sentiment(sentence)[1] for sentence in sent_tokenize(body)]) 

def determine_post_length(body):
	""" This function determines and returns the length of the provided input string in sentences.
	It uses the nltk sent_tokenize function to do so. """
	return(len(sent_tokenize(body)))

def determine_sentence_length(body):
	""" This function determines and returns the average length of the sentences in the provided input string in words.
	It uses the nltk word_tokenize function to do so. """
	return np.mean([len(word_tokenize(sentence)) for sentence in sent_tokenize(body)])

#---------------------------
# Put a specific column in front
#---------------------------
def swap_to_front(header, column):
	""" this function takes a list of column names, and the column name that is to be placed in front. It places this column name on the [1]th position of the list of column names. """
	header.remove(column)
	header.insert(0,column)
	return header

def place_feature_first(df,col_name="Date & Time"):
	""" this function takes a dataframe, and the name of the column that is to be switched to the front. It places this column on the [1]th position in the dataframe. """
	header = df.columns.tolist()
	header = swap_to_front(header,col_name)
	new_df = df.reindex(columns = header)
	return new_df

############################################################
## MAIN CODE
############################################################
# get files
posts = json.load(open('posts.json'))
users = json.load(open('users.json'))

# path for saving csv files
path_out = r"... \user-csvs"

# sort out the data per user in dictionaries (values are either Posts or Times in lists), and make a list of all Datetimes in the data
P,T,D = make_P_T_and_D(topics1,topics2,posts1,posts2)

# filter some users out
over_treshold = determine_active_users(['76']) # determines all users that have posted at least 30 posts, from a specified list or (default) from all users in the userstatus file.

# chronological list of bin boundaries
binlist = make_binlist(D,days=1)
first_access_dict = make_first_access_dict()
last_access_dict = make_last_access_dict()

for user in over_treshold:
	# initiate some user-specific variables
	first_access = datetime.strptime(first_access_dict[user], '%d/%m/%Y - %H:%M')
	last_access = datetime.strptime(last_access_dict[user], '%d/%m/%Y - %H:%M')
	first_date = 0 # different from first access: people may not post directly after making an account
	last_date = 0
	inactivity = 0 # later changed to first_access and updated in every next time bin
	
	# all lists starting with csv_ are lists that will eventually contain all values that end up in the csv file
	csv_date = []			#datetime
	csv_sentiment = []		#sentiment value (-1 to 1)
	csv_questionmarks = []	#question mark-ending sentences ratio (float)
	csv_subjectivity = []	#subjectivity value (0 to 1)
	csv_sentencelength = []	#length of sentence in words (float)
	csv_postlength = []		#length of post in sentences (float)
	csv_inactivity = []		#hours passed since last activity (int)
	csv_churn_1month = []	#dependent variable: churn in one month
	csv_churn_2months = []	#dependent variable: churn in two months
	csv_churn_3months = []	#dependent variable: churn in three months
							#retrospective variables are determined in a separate step of the algorithm
	
	# dictionaries to keep track of activity within certain time bins
	bindict = defaultdict(list)		#bins with size 'timetick' (default 1h), values = post-times 
	postdict = defaultdict(list)	#bins with size 'timetick' (default 1h), values = posts
	churn_1monthdict = defaultdict(list)
	churn_2monthsdict = defaultdict(list)
	churn_3monthsdict = defaultdict(list)
	
	for index in range(len(tqdm(binlist))):
		if index==len(binlist):
			break
		else:
			lower = binlist[index]
			upper = binlist[index+1]
			
			# Loop through T, annotating all posts for the selected user
=			for time in T[user]:
				# if the time stamp is in this time bin, then....
				if lower<=datetime.strptime(time, '%d/%m/%Y - %H:%M')<upper:
					bindict[lower,upper].append(time)
					postdict[lower,upper].append(P[user][T[user].index(time)]) 

					# determine the first and last active dates
					if first_date == 0:
						first_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')
						last_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')
					else:
						last_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')


					# determine whether the patient will churn within one / two / three months
					churn_1monthdict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=1))>last_access else 0)
					churn_2monthsdict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=2))>last_access else 0)
					churn_3monthsdict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=3))>last_access else 0)

			if len(bindict[lower,upper])==0:
				bindict[lower,upper]=[]
				postdict[lower,upper]=[]
				
			# Fill csv_feature-lists with values 
			# concatenate all posts in one time bin
			body = '. '.join(postdict[lower,upper]) #can be empty!
		
			#inactivity is 0 if before first_access
			if datetime.strptime(time, '%d/%m/%Y - %H:%M')<first_access:
				inactivity = 0
			# after first_access, inactivity increases with one in every empty time bin
			elif len(body) == 0:
				inactivity+=1
			# after first access, inactivity resets in filled time bin and other features are annotated as well
			else:	
				csv_date.append(lower)
				csv_sentiment.append(determine_sentiment(body))
				csv_questionmarks.append(determine_questionmarks(body))
				csv_subjectivity.append(determine_subjectivity(body))
				csv_sentencelength.append(determine_sentence_length(body))
				csv_postlength.append(np.mean([determine_post_length(x) for x in postdict[lower,upper]]))
				csv_inactivity.append(inactivity)
				inactivity = 0
				
				csv_churn_1month.append(1 if np.mean(churn_1monthdict[lower,upper])>=0.5 else 0)
				csv_churn_2months.append(1 if np.mean(churn_2monthsdict[lower,upper])>=0.5 else 0)
				csv_churn_3months.append(1 if np.mean(churn_3monthsdict[lower,upper])>=0.5 else 0)
				
	# Save the results for this user in a csv file
	df = pd.DataFrame({"Date & Time": csv_date, "Sentiment": csv_sentiment, "Questions": csv_questionmarks, 
					   "Subjectivity": csv_subjectivity, "Words/Sentence": csv_sentencelength, 
					   "Sentences/Post": csv_postlength, "Inactivity": csv_inactivity,
					   "Churn in 1m": csv_churn_1months,"Churn in 2m": csv_churn_2months,"Churn in 3m": csv_churn_3months})
	# used to control column order:
	df = place_feature_first(df)
	
	# set name of csv file and save file in a separate folder
	name = "features_user_"+str(user)+".csv"	   
	df.to_csv(os.path.join(path_out,name),index=False)