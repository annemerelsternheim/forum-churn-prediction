# this file generates the list of tokens in the data, that will be used for improvement of the pattern sentiment miner
# another file is supposed to generate the variables that will be used for machine learning.


###############################################
# ------------------ imports ------------------
###############################################

import json, re, time, unicodedata, unidecode, codecs, random
from pprint import pprint
from lxml import etree
from pattern.nl import parse, split, parsetree
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


###############################################
# ------------------- files -------------------
###############################################

topics = json.load(open('D:\\4. Data\\Amazones_Forum_Export_JSON\\2017-12-07T13-36-51_amazones_forum_topics_export.json'))
posts = json.load(open('D:\\4. Data\\Amazones_Forum_Export_JSON\\2017-12-07T13-39-20_amazones_forum_posts_export.json'))


###############################################
# ----------------- functions -----------------
###############################################


# --------- make count of all tokens in the tokendict  ---------
# expects a dictionary with POS tags as keys, and word tokens (lowercase) as values.
# returns a dict with POS as key, and (token, count) as values
def make_tokentypedict(tokendict):
		tokentypedict = defaultdict(list)
		for POS in tokendict:
			typedict = defaultdict(list)
			for token in tokendict[POS]:
				typedict[token].append(1)
			for types in typedict:
				tokentypedict[POS].append((types,len(typedict[types])))
		return(tokentypedict)
		

# --------- sort items of the tokentypedict  ---------
# expects the .items() of a dictionary with POS as key, and (token,count) as values
# returns a list of (token,count), sorted by count
def sort_by_occurence(dictionaryitems):
		sortedlist = []
		for POS in dictionaryitems:
			name=POS[0]
			words = POS[1] #list with tuples. This can be sorted
			sortedwords = sorted(words, key=lambda tup: tup[1],reverse=True)
			newPOS=(name,sortedwords)
			sortedlist.append(newPOS)
		return sortedlist