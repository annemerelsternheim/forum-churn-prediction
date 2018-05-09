# coding: utf-8

# python <file>.py <topics>.json <posts>.json <path-for-csv-files>

# optional: adding one or more user-IDs. This enables you to run the program for just a few users, instead of the entire batch.
# non-existing user-IDs will not raise an error, they are simply ignored

# command line example:
# python 2018_03_29_variables_to_csv.py "D:\topics.json" "D:\posts.json" C:\user-csvs 1144,1433

# a successful run will within a few seconds show a progress bar (which takes 40 seconds to complete, on my computer): this progress bar visualises the colleciton of the data
# then you will see a second progress bar, showing how many user IDs were selected to analyse data from, and an estimation of the total running time (at least after the first user is analysed)
# a third progress bar shows how long the analysis for the first user is taking.
# Then some info about this user's activity is printed
# and then the next, and the next, and the next user...
# a square is printed if the program has finished. For fun.

# quick copy-paste for me:
# python 2018_03_29_variables_to_csv.py "D:\4. Data\Amazones_Forum_Export_JSON\2017-12-07T13-36-51_amazones_forum_topics_export.json" "D:\4. Data\Amazones_Forum_Export_JSON\2017-12-07T13-39-20_amazones_forum_posts_export.json" "C:\Users\sternheimam\Desktop\my-notebook\user-csvs" 1144

############################################################
## IMPORTS
############################################################

import json, re, time, unicodedata, unidecode, itertools, os, sys, glob
from collections import defaultdict
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from pattern.nl import sentiment
import pandas as pd
from tqdm import *


############################################################
## FUNCTION DEFINITIONS
############################################################

#---------------------------
# Prepare text in .json files for measurements etc
#---------------------------
def remove_non_ascii(text):
	""" this function expects a string, and removes non-ascii characters from it """
	return unidecode.unidecode(text)
	
def cleanup(text):
	""" this function expects a string (post from the BVN/Amazones forum), and returns a cleaner version of it """
	# remove all links, images, quotes, and emailaddresses
	text=re.sub('<a.*?>(.*?)</a>','',text) #remove links
	text=re.sub('(http:|www)\S*','',text) #remove links without markup
	text=re.sub('\[\\\/url\]','',text)
	text=re.sub('<img.*?/>', '',text) #remove images
	text=re.sub('<div class="bb-quote">((\s|\S)*?)</div>','',text) #remove quotes
	text=re.sub('<script.*?>([\S\s]*?)</script>','',text) #remove emailaddresses

	# replace all emoticon-icons
	text=re.sub('<img.*?title="(.*?)".*?/>', '(EMO:\\1)',text) #replace emoticons by textual indicators 

	# replace (most) sideways latin emoticons
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
	# :s
	# :x is eigenlijk geen kus, geloof ik...

	#other important adjustments:
	text=re.sub('m\'?n\s','mijn ',text) # replacing m'n and mn with mijn, so it gets parsed correctly.
	text=re.sub('z\'?n\s','zijn ',text) #replacing z'n and zn with zijn
	text=re.sub('d\'?r\s','haar ',text) #replacing d'r and dr with zijn (only if followed by space, so dr. stays dr.)

	# replace all emoticons (and other things) written between double colons
	text=re.sub(':([a-zA-Z]+):','(EMO:\\1)',text)

	# remove remaining markup
	text=re.sub('</?(ol|style|b|p|em|u|i|strong|br|span|div|blockquote|li)(.*?)/?>','',text)
	text=re.sub('(\[|\]|\{|\})', '',text)

	# separate text from punctuation (may cause double/triple spaces - does not matter at this point)
	text = re.sub('(\.{2,}|/|\)|,|!|\?)','\\1 ',text) # space behind
	text = re.sub('(/|\()',' \\1',text) # space in front
	text = re.sub('(\w{2,})(\.|,)','\\1 \\2 ',text) #space 'between'

	return(remove_non_ascii(text))


#---------------------------
# Go through data, save differently
#---------------------------
def make_P_T_and_D(topics1,topics2,posts1,posts2):
	""" this function takes the .json files containing the thread starts and responses, and returns three things:
	[0]: a dictionary with the user-ID as key, and the post as value;
	[1]: a dictionary with the user-ID as key, and the time of posting as a value;
	[2]: a list of all datetimes present in the data (sorted by date, because the .json was already sorted) """
	P = defaultdict(list)
	T = defaultdict(list)
	D = []

	print "Loading ..."
	with tqdm(total=len(topics1)+len(topics2)+len(posts1)+len(posts2)) as pbar:
		for t1 in reversed(topics1):
			pbar.update(1)
			P[t1['Author uid']].append((cleanup(t1["Body"]),1))
			T[t1['Author uid']].append(t1['Post date'])
			D.append(datetime.strptime(t1['Post date'], '%d/%m/%Y - %H:%M'))
			
		for t2 in reversed(topics2):
			pbar.update(1)
			P[t2['user_id']].append((cleanup(t2["body"]),1))
			T[t2['user_id']].append(t2['post_date'])
			D.append(datetime.strptime(t2['post_date'], '%d/%m/%Y - %H:%M'))

		for p1 in reversed(posts1):
			pbar.update(1)
			P[p1['Auteur-uid']].append((cleanup(p1["Body"]),0))
			T[p1['Auteur-uid']].append(p1['Datum van inzending'])
			D.append(datetime.strptime(p1['Datum van inzending'], '%d/%m/%Y - %H:%M'))
	
		for p2 in reversed(posts2):
			pbar.update(1)
			P[p2['user_id']].append((cleanup(p2["body"]),0))
			T[p2['user_id']].append(p2['post_date'])
			D.append(datetime.strptime(p2['post_date'], '%d/%m/%Y - %H:%M'))
	print "Loading complete! Now loop through all users, and make csv files: "
	return (P,T,D)

#---------------------------
# Make list of bin-bondaries-to-be (binlist)
#---------------------------
def make_binlist(D,hours = None, days = None, months = None):
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

	# als er uren ingegeven zijn, neem dan een zoveel-uur-bin
	if hours != None:
		return([lower + timedelta(hours=x) for x in range(0, 24*((upper-lower).days), hours)])
	# als er dagen ingegeven zijn, neem dan een zoveel-dag-bin
	elif days != None:
		return([lower + timedelta(days=x) for x in range(0, (upper-lower).days, days)])
	# als er maanden ingegeven zijn, neem dan een zoveel-maand-bin
	elif months != None:
		return([lower + relativedelta(months=x) for x in range(0, ((upper-lower).days)/30, months)])
	else:
		print "please specify the size of the bins in hours, days or months"
		return []

#---------------------------
# List all created dates / first accesses
#---------------------------
def make_first_access_dict(first_access_dict = dict()):
	for u in userstatus:
		first_access_dict[u['user_id']] = u['created_date']
	return first_access_dict

#---------------------------
# List all last accesses (for both blocked and active users)
#---------------------------
def make_last_access_dict(last_access_dict = dict()):
	for u in userstatus:
		last_access_dict[u['user_id']] = u['last_access']
	return last_access_dict
#---------------------------
# User selection
#---------------------------
def determine_active_users(include_only = []):
	global over_treshold
	
	E = open('exclude.txt', 'r')
	lines = E.readlines()
	exclude = [l.rstrip() for l in lines]
	
	# if the function is given an argument, work with that specific data instead of all users
	if include_only == []:
		userlist = [u['user_id'] for u in userstatus]
	else:
		userlist = include_only

	# only consider users with a minimum of 30 posts, and those whose data was not obviously messed up/with
	for user in userlist:
		if len(T[user])<30:
			pass
		elif user in exclude:
			pass
		else:
			over_treshold.append(user)

#---------------------------
# Variable annotation
#---------------------------
def determine_questionmarks(body, Q=0):
	""" This function counts and returns the number of sentences in the provided input string
	that ends in at least one question mark """
	for sentence in sent_tokenize(body):
		if re.search('\?+', sentence):
			Q+=1
	if len(sent_tokenize(body))!=0:
		return float(Q)/float(len(sent_tokenize(body)))
	else:
		return 0

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
	#word_tokenize also considers interpunction a word
	return np.mean([len(word_tokenize(sentence)) for sentence in sent_tokenize(body)])

def determine_week_activity(first_date,last_date,bindict):
	""" This function returns a dictionary that has kept track of the activity in week-bins, instead of day-bins.
	It expects two dates, to indicate in between which dates the dictionary should be built,
	and expects a dictionary in which all the user's active times are already stored"""
	for d in range(0, (last_date-first_date).days,7):
		week_start = first_date+timedelta(days=d)
		week_end = first_date+timedelta(days=d+7)
		for date in list(itertools.chain.from_iterable(bindict.values())):
			if week_start<=datetime.strptime(date, '%d/%m/%Y - %H:%M')<week_end:
				weekcountdict[week_start,week_end].append(1)
		if len(weekcountdict[week_start,week_end])==0:
			weekcountdict[week_start,week_end] = []
	return weekcountdict

def determine_past_activity(bindict,index,hours_back=24):
	""" This function returns the number of times a user has been active in the last hours_back hours (default 24h).
	The first hours_back time bins will for now have a value of 0 by default, to keep things easy."""
	if 0<= index-hours_back<=len(bindict):
		past_activity = np.sum([len(bindict[binlist[index-(x+1)],binlist[index+1-(x+1)]]) for x in range(hours_back)])
	else:
		past_activity=0
	return past_activity

def print_information(user, T, first_date,last_date,weekcount):
	""" this function prints useful basic information on the user's activity.
	It needs quite some input so make sure you've got them all:
	1) user ID, 2) a dictionary containing users-IDs as key, and any activity log as value,
	3,4) the first and last date of activity, and 5) the dictionary that kept track of the week activity. """
	print "User:", user
	print "posted one or more posts in", len(T[user]), "'bins'." 
	print "The first post: ", first_date
	print "The last post: ", last_date
	print "Activity spread over: ", last_date-first_date
	print "The average nr of posts per week: ", np.mean([len(x) for x in weekcount.values()]), "including long times of inactivity."
	print "The average nr of posts in non-empty weeks: ", np.mean([len(x) for x in weekcount.values() if not x==[]])
	print "The range of activity: ", min([len(x) for x in weekcount.values()]), " to ", max([len(x) for x in weekcount.values()]), " posts per week"
	print

#---------------------------
# Kneading df into the right shape
#---------------------------
def swap_to_front(header, dependent_variable):
	""" this function takes a list of column names, and the column name that is to be placed in front. It places this column name on the [1]th position of the list of column names. """
	header.remove(dependent_variable)
	header.insert(0,dependent_variable)
	return header

def place_datetime_first(df,dependent_variable="Date & Time"):
	""" this function takes a dataframe, and the name of the column that is to be switched to the front. It places this column on the [1]th position in the dataframe. """
	header = df.columns.tolist()
	header = swap_to_front(header,dependent_variable)
	new_df = df.reindex(columns = header)
	return new_df

############################################################
## MAIN CODE
############################################################
#---------------------------
# load the files
#---------------------------
topics1 = json.load(open('D:\\4. Data\\Amazones\\oud_topics.json'))
topics2 = json.load(open('D:\\4. Data\\Amazones\\nieuw_topics.json'))
posts1 = json.load(open('D:\\4. Data\\Amazones\\oud_posts.json'))
posts2 = json.load(open('D:\\4. Data\\Amazones\\nieuw_posts.json'))
users1 = json.load(open('D:\\4. Data\\Amazones\\oud_users.json'))
users2 = json.load(open('D:\\4. Data\\Amazones\\nieuw_users.json'))
userstatus = json.load(open('D:\\4. Data\\Amazones\\nieuw_users_status.json'))

#---------------------------
# sort out the data per user (values are either Posts or Times in lists), and make a list of all datetimes in the data
#---------------------------
P,T,D = make_P_T_and_D(topics1,topics2,posts1,posts2)

#---------------------------
# determine 'relevant' users (users with at least 30 posts, which were hopefully not influenced by the technical issues)
#---------------------------
# initiate empty list of users relevant to measure
over_treshold = []
determine_active_users(['76']) # determines all users that have posted at least 30 posts, from a specified list or (default) from all users in the userstatus file.
#---------------------------
# global variables
#---------------------------
# path for saving csv files
path = r"C:\Users\sternheimam\Desktop\my-notebook\user-csvs"

# chronological list of (lower) bin boundaries
binlist = make_binlist(D,days=1)
first_access_dict = make_first_access_dict()
last_access_dict = make_last_access_dict()

# The negative difference for measuring the 'backtrack' feature
neg_diff = 24

#---------------------------
# go through all users in over_treshold, and do stuff..
#---------------------------
# show the progress, while going through the active users
with tqdm(total=len(over_treshold)) as processbar:
	for user in over_treshold:
		processbar.update(1)
		
		#---------------------------
		# initiate some user-specific variables
		#---------------------------
		first_access = datetime.strptime(first_access_dict[user], '%d/%m/%Y - %H:%M')
		last_access = datetime.strptime(last_access_dict[user], '%d/%m/%Y - %H:%M')
		first_date = 0
		last_date = 0
		inactivity = 0 # change to first access date, then re-calculate first inactivity with this date
		
		# all lists starting with csv_ are lists that will eventually contain all values that end up in the csv file
		csv_date = []			#datetime
		csv_sentiment = []	   #sentiment value (-1 to 1)
		csv_questionmarks = []   #question mark-ending sentences (float)
		csv_subjectivity = []	#subjectivity value (0 to 1)
		csv_sentencelength = []  #length of sentence in words (float)
		csv_postlength = []	  #length of post in sentences (float)
		csv_startposts = []	  #1 for a thread start, 0 for a response (float) 
		csv_inactivity = []	  #hours passed since last activity (int)
		csv_backtrack = []	   #posts posted in last x hours (x = neg_diff; default 24h)
		csv_churn_year = []
		csv_churn_6months = []
		csv_churn_3months = []
		
		# dictionaries to keep track of activity within certain time bins
		bindict = defaultdict(list)		#bins with size 'timetick' (default 1h), values = post-times 
		postdict = defaultdict(list)	   #bins with size 'timetick' (default 1h), values = posts
		metadict = defaultdict(list)	   #bins with size 'timetick' (default 1h), values = 1 or 0 (start or response)
		weekcountdict = defaultdict(list)  #bins with size = 7 days, values = '1' for every post
		backtrackdict = defaultdict(list)  #bins with size = neg_diff (default 24h), values = nr of posts in last neg_diff hours
		churn_yeardict = defaultdict(list)
		churn_6monthsdict = defaultdict(list)
		churn_3monthsdict = defaultdict(list)
				# TO DO: linguistic markers, like adjectives / pronouns / emoticons, and the diversity of topics / vocabulary
		
		#---------------------------
		# loop through the (sorted) list of datetimes, and do stuff..
		#---------------------------
		for index,boundary in enumerate(tqdm(binlist)):
			# determine time bin boundaries for dictionaries
			if index+1>=len(binlist):
				break
			else:
				lower = binlist[index]
				upper = binlist[index+1]
				
				#---------------------------
				# Loop through T, collecting all activity for the selected user
				#---------------------------
				# determine in which time bin the user's activity belongs
				for time in T[user]: # T contains more users than userstatus, but that does not matter because the users not in userstatus are simply ignored in T.
					if lower<=datetime.strptime(time, '%d/%m/%Y - %H:%M')<upper:
						bindict[lower,upper].append(time)
						
						# and determine how active the user has been in past neg_diff hours
						past_activity = determine_past_activity(bindict,index,neg_diff)
						backtrackdict[lower,upper].append(past_activity)
						
						# determine the first and last active dates
						if first_date == 0:
							first_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')
							last_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')
						else:
							last_date = datetime.strptime(time, '%d/%m/%Y - %H:%M')
						
						# split the text in P from the start/response-information
						body = P[user][T[user].index(time)][0]
						meta = P[user][T[user].index(time)][1]
						postdict[lower,upper].append(body) 
						metadict[lower,upper].append(meta)

						# determine whether the patient will churn within one year/6 months/3 months
						churn_yeardict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=12))>last_access else 0)
						churn_6monthsdict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=6))>last_access else 0)
						churn_3monthsdict[lower,upper].append(1 if (datetime.strptime(time, '%d/%m/%Y - %H:%M')+relativedelta(months=3))>last_access else 0)

				# fill up the still-empty places in the dictionary
				if len(bindict[lower,upper])==0:
					bindict[lower,upper]=[]
					postdict[lower,upper]=[]
					metadict[lower,upper]=[]
					backtrackdict[lower,upper]=[]

					
				#---------------------------
				# Fill csv_feature-lists with values 
				#---------------------------				
				# Treat different posts within same bin 'as one' (concatenate them)
				body = '. '.join(postdict[lower,upper]) #can be empty!
			
				#when the bin is before the first access date, you don't do anything. Inactivity is kept 0 until bins are 'within' the user's frame
				if datetime.strptime(time, '%d/%m/%Y - %H:%M')<first_access:
					inactivity = 0
				# when the bin is after last acces AND the bin is empty, add 1 to the inactivity feature (this keeps track of how many bins were between last activity and current)
				elif len(body) == 0:
					inactivity+=1
				# when the bin is after last access AND NOT empty, append a value to all csv_feature-lists (and reset 'inactivity')
				else:	
					csv_date.append(lower)
					csv_sentiment.append(determine_sentiment(body))
					csv_questionmarks.append(determine_questionmarks(body))
					csv_subjectivity.append(determine_subjectivity(body))
					csv_sentencelength.append(determine_sentence_length(body))
					csv_postlength.append(np.mean([determine_post_length(x) for x in postdict[lower,upper]]))
					csv_startposts.append(np.mean(metadict[lower,upper]))
					csv_inactivity.append(inactivity)
					csv_backtrack.append(np.sum(backtrackdict[lower,upper][-1]))
					csv_churn_year.append(1 if np.mean(churn_yeardict[lower,upper])>=0.5 else 0)
					csv_churn_6months.append(1 if np.mean(churn_6monthsdict[lower,upper])>=0.5 else 0)
					csv_churn_3months.append(1 if np.mean(churn_3monthsdict[lower,upper])>=0.5 else 0)
					inactivity = 0
					
		#---------------------------
		# Report the results for this user
		#--------------------------- 
		# determine average activity: over active period, and over only-active weeks 
		#weekcount = determine_week_activity(first_date,last_date,bindict)   
		#print_information(user, T, first_date,last_date,weekcount)			
				
		# put all csv_features into a dataframe
		df = pd.DataFrame({"Date & Time": csv_date, "Sentiment": csv_sentiment, "Questions": csv_questionmarks, 
						   "Subjectivity": csv_subjectivity, "Words/Sentence": csv_sentencelength, 
						   "Sentences/Post": csv_postlength, "First posts": csv_startposts,
						   "Inactivity": csv_inactivity, "Posts in last 24H": csv_backtrack,
						   "Churn in y": csv_churn_year,"Churn in 6m": csv_churn_6months,
						   "Churn in 3m": csv_churn_3months})#.dropna()
		df = place_datetime_first(df)
		name = "features_user_"+str(user)+".csv"	   
		df.to_csv(os.path.join(path,name),index=False)
print " _ "
print "|_|"