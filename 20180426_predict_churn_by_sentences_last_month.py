# coding: utf-8

############################################################
## IMPORTS
############################################################
import glob, os, re, sys
import pandas as pd
from tqdm import *
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

############################################################
## FUNCITON DEFINITIONS
############################################################

	
############################################################
## MAIN CODE
############################################################

#---------------------------
# Fetch all .csv-files in specified path, and restructure the information in it in a useful way
#---------------------------
# step2: make 3-month churn the dependent variable, and 'sentences per post' the indepenent variable

path_in = r'C:\Users\sternheimam\Desktop\my-notebook\user-csvs'
path_out = r'C:\Users\sternheimam\Desktop\my-notebook\user-csvs_sentences'

for filename in tqdm(glob.glob(os.path.join(path_in, '*.csv'))):
	user = re.search("_(.[0-9]*)\.csv", filename).group(1)
	last_active_bin = 0 #should be the date of subscription
	new_sentences = [] # next step could be to multiply this by the word count
	df = pd.read_csv(filename)
	dates = df['Date & Time']
	sentences = df['Sentences/Post']
	first = min([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates])
	last = max([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates])
	binlist = [first+(bin*timedelta(days=1)) for bin in range((last-first).days)] 
	earlier = [bin+relativedelta(months=-1) for bin in binlist]
	
	for bin in tqdm(binlist):
		datesDT = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
		if bin in datesDT:
			new_sentences.append(sentences[datesDT.index(bin)])
			last_active_bin = bin
		else:
			new_sentences.append(0)
	df = pd.DataFrame({"Date & Time": binlist, "Sentences": new_sentences})
	
	binlist = df['Date & Time']
	sentences = df['Sentences']
	first = min([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for bin in binlist])
	last = max([datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for bin in binlist])
	earlier = [bin+relativedelta(months=-1) for bin in binlist]
	Values = []

	# loop nu door de hele lijst. voeg aan bin x de waarde van x-1 tm x-29 toe
	for i,bin in enumerate(binlist):
		values = []
		for j in range((binlist[i]-earlier[i]).days):
			if i-j >= 0:
				values.append(sentences[i-j])
		Values.append(np.sum(values))
	
	df['Over past 30 days'] = Values
	name = str(user)+"_churn-sentences.csv"
	df.to_csv(os.path.join(path_out,name),index=False)