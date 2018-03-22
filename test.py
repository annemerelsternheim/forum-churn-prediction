#!/usr/bin/python
# This Python file uses the following encoding: utf-8

# this file is supposed to generate the variables that will be used for machine learning.
# another file generates the list of tokens in the data, that will be used for improvement of the pattern sentiment miner


###############################################
# ------------------ imports ------------------
###############################################

# imports
import json, re, time, unicodedata, unidecode, codecs, random, math, warnings
from pprint import pprint
from lxml import etree
from pattern.nl import parse, split, parsetree
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from pattern.nl import sentiment


###############################################
# ------------------- files -------------------
###############################################

topics = json.load(open('D:\\4. Data\\Amazones_Forum_Export_JSON\\2017-12-07T13-36-51_amazones_forum_topics_export.json'))
posts = json.load(open('D:\\4. Data\\Amazones_Forum_Export_JSON\\2017-12-07T13-39-20_amazones_forum_posts_export.json'))


###############################################
# ----------------- functions -----------------
###############################################

# --------- remove non-ascii characters from the file  ---------
# expects a string as input
# returns a string as output: non-ascii characters are converted to their closest ascii-relative. if that was not possible, the character is removed.
def remove_non_ascii(text):
		return unidecode.unidecode(text)
		return ''.join([i if ord(i) < 128 else ' ' for i in text])

# --------- remove links, images, quotes, replace emoticons, etc  ---------
# expects a string as input
# returms a string as output: very readable, emoticons replaced by 'markers'.  string does not contain any characters any more that ought not to be parsed.
def cleanup(text):
		#remove all links, images, quotes, and emailaddresses
		text=re.sub('<a.*?>(.*?)</a>','',text) #remove links
		text=re.sub('(http:|www)\S*','',text) #remove links without markup
		text=re.sub('\[\\\/url\]','',text)
		text=re.sub('<img.*?/>', '',text) #remove images
		text=re.sub('<div class="bb-quote">((\s|\S)*?)</div>','',text) #remove quotes
		text=re.sub('<script.*?>([\S\s]*?)</script>','',text) #remove emailaddresses

		#replace all emoticon-icons
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
		text=re.sub('(/|\()',' \\1',text) # space in front
		text=re.sub('(\w{2,})(\.|,)','\\1 \\2 ',text) #space 'between'
		
		return(text)	

		
# --------- collect time information  ---------
# expects two json files, which contain text and dates
# returns a tuple of three: two dictionaries with uid as key, and [0] posts as values, and [1] times as values. and a list of all dates in the data (as datetime.datetime objects)
 
def make_P_T_and_D(topics,posts,count=2500):
		P = defaultdict(list)
		T = defaultdict(list)
		D = []

		for t in reversed(topics):    
			P[t['Author uid']].append(remove_non_ascii(cleanup(t["Body"])))
			T[t['Author uid']].append(t['Post date'])
			D.append(datetime.strptime(t['Post date'], '%d/%m/%Y - %H:%M'))

			count-=1
			if count-1<=0:
				break
			for p in reversed(posts):
				if p['Forum Topic ID'] == t['Nid']:
					P[p['Auteur-uid']].append(remove_non_ascii(cleanup(p["Body"])))
					T[p['Auteur-uid']].append(p['Datum van inzending'])
					D.append(datetime.strptime(p['Datum van inzending'], '%d/%m/%Y - %H:%M'))

					count-=1
					if count-1<=0:
						break
		return (P,T,D)
	
# --------- make a list of time bins (timetick default = 24h)  ---------
# expects a list of datetime.datetime objects
# returns a list of datetime.datetime objects, 24h apart (or different if explicitly indicated), which starts at the first 4:00 before, and ends at the first 4:00 after the lowest and highest values in the input list (the last bin will include, and go beyond this 'upper' value, if timetick >1)
def make_binlist(D,timetick=1):
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

		return([lower + timedelta(days=x) for x in range(0, (upper-lower).days, timetick)])
		

# --------- keep track of variables in the data ---------
# expects a string
# returns an int, representing the nr of sentences in the string that ended in a question mark
def determine_questionmarks(body):
		Q = 0
		for sentence in sent_tokenize(body):
			if re.search('\?+', sentence):
				Q+=1
		if len(sent_tokenize(body))!=0:
			return float(Q)/float(len(sent_tokenize(body)))
		else:
			return 0
			
# expects a string
# returns an int, representing the average sentiment score per sentence in the string, calculated by pattern
def determine_sentiment(body):
		S = []
		for sentence in sent_tokenize(body):
			S.append(sentiment(sentence)[0])
		return np.mean(S)

# expects a string
# returns an int, representing the average objectivity score per sentence in the string, calculated by pattern
def determine_objectivity(body):
		O = []
		for sentence in sent_tokenize(body):
			O.append(sentiment(sentence)[1])
		return np.mean(O)
		
# expects a string
# returns an int, representing the total nr of sentences the string is built of
def determine_length(body):
		# in sentences:
		return(len(sent_tokenize(body)))

# expects a string
# returns an int, representing the average nr of words in sentences that occur in the string
def determine_sent_length(body):
		# in words:
		L = []
		for sentence in sent_tokenize(body):
			L.append(len(word_tokenize(sentence)))
		return np.mean(L)
				
# --------- present variables in a plot ---------
# expects all 'mean' dictionaries
# returns a plot for every user, which makes it easy to compare variables through time

def compare_variables(pp,user,nr_of_posts,mean_quest,mean_object,mean_sents,mean_length,mean_sent_length):
		fig = plt.figure(1)
		ax = plt.subplot(111)
		ax.set_position([0.055,-0.5,0.85,0.8])

		posts, = ax.plot(nr_of_posts.values(),'b^', label = 'nr of posts', alpha = 0.5) #blue
		senlength, = ax.plot(mean_sents_length.values(),'k^', label = 'mean sentence length (words)',alpha = 0.5) #black
		postlength, = ax.plot(mean_length.values(),'c^', label = 'mean post length (sentences)',alpha = 0.5) #cyan

		first_legend = plt.legend(handles=[posts,senlength,postlength], title = "left axis", loc='upper left', bbox_to_anchor=(0, -0.1),ncol=1)
		axx = plt.gca().add_artist(first_legend)
		
		ax1 = ax.twinx()
		ax1.set_position([0.055,-0.5,0.85,0.8])

		qmarks, = ax1.plot(mean_quest.values(),'g.', label = 'question ratio',alpha = 0.5) #green
		ovalues, = ax1.plot(mean_object.values(),'r.', label = 'objectivity',alpha = 0.5) #red
		svalues, = ax1.plot(mean_sent.values(),'m.', label = 'sentiment',alpha = 0.5) #magenta

		second_legend = plt.legend(handles=[qmarks,ovalues,svalues], title = "right axis",loc='upper right', bbox_to_anchor=(1, -0.1),ncol=1)

		ax.set_ylim(0,40)
		ax1.set_ylim(-1,1)
		
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height, box.width, box.height * 0.7])
		box = ax1.get_position()
		ax1.set_position([box.x0, box.y0 + box.height, box.width, box.height * 0.7])
		
		plt.title(user)
		plt.show()
		pp.savefig(fig, dpi = 300, transparent = True)

###############################################
# -------------------- code -------------------
###############################################

PTD = make_P_T_and_D(topics,posts) 
P = PTD[0]
T = PTD[1]
D = PTD[2]

plt.close()
binlist = make_binlist(D,7) #timetick in (whole) days
pp = PdfPages("plots-author-BVN.pdf")

for user in T:
    # variables
    bindict = defaultdict(list)
    postdict = defaultdict(list)
    sendict = defaultdict(list)
    questdict = defaultdict(list)
    objecdict = defaultdict(list)
    lengthdict = defaultdict(list)
    senlengthdict = defaultdict(list)
    # length of posts (in words, or sentences)
    # nr of replies to posts vs nr of starting posts
    # linguistic markers, like adjectives / pronouns / emoticons, and the diversity of topics / vocabulary

    # variables (plottable)
    nr_of_posts = dict()
    mean_quest = dict()
    mean_sent = dict()
    mean_object = dict()
    mean_length = dict()
    mean_sents_length = dict()

    for index,boundary in enumerate(binlist):
        if index+1>=len(binlist):
            break
        else:
            lower = binlist[index]
            upper = binlist[index+1]

            for time in T[user]:
                if lower<=datetime.strptime(time, '%d/%m/%Y - %H:%M')<upper:
                    body = P[user][T[user].index(time)]
                    senti = determine_sentiment(body) # average sentiment per sentence in body
                    questionmarks = determine_questionmarks(body) # ratio of sentences in body ending with question marks
                    objectivity = determine_objectivity(body)
                    length = determine_length(body)
                    sentence_length = determine_sent_length(body)

                    bindict[lower,upper].append(time)
                    postdict[lower,upper].append(body) 
                    sendict[lower,upper].append(senti)
                    questdict[lower,upper].append(questionmarks)
                    objecdict[lower,upper].append(objectivity)
                    lengthdict[lower,upper].append(length)
                    senlengthdict[lower,upper].append(sentence_length)

        # fill up empty places in dictionary
        # is this necessary? I convert to mean dicts either way...?
        # perhaps only for ML purposes
        if len(bindict[lower,upper])==0:
            bindict[lower,upper]=[]
            postdict[lower,upper]=[]
            sendict[lower,upper]=[]
            questdict[lower,upper]=[]
            objecdict[lower,upper]=[]
            lengthdict[lower,upper]=[]
            senlengthdict[lower,upper]=[]

    #convert dictionaries to things you want plotted, like averages:
    for lower,upper in bindict:
        # mean nr of question sentences per timetick
        if len(bindict[lower,upper])==0:
            nr_of_posts[lower,upper]=float('nan')
            mean_quest[lower,upper]=float('nan')
            mean_sent[lower,upper]=float('nan')
            mean_object[lower,upper]=float('nan')
            mean_length[lower,upper]=float('nan')
            mean_sents_length[lower,upper]=float('nan')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                nr_of_posts[lower,upper]=len(bindict[lower,upper])
                mean_quest[lower,upper]=np.mean(questdict[lower,upper])
                mean_sent[lower,upper]=np.mean(sendict[lower,upper])
                mean_object[lower,upper]=np.mean(objecdict[lower,upper])
                mean_length[lower,upper]=np.mean(lengthdict[lower,upper])
                mean_sents_length[lower,upper]=np.mean(senlengthdict[lower,upper])
                
    # here we start making plots, because we have collected all data for a single user, and now we plot this:
    if len([x for x in nr_of_posts.values() if not math.isnan(x)])< 30:
        pass
    else:
        print len([x for x in nr_of_posts.values() if not math.isnan(x)])
        compare_variables(pp,user,nr_of_posts,mean_quest,mean_object,mean_sent,mean_length,mean_sents_length)
pp.close()

##### maak statistische correlaties tussen de curves
##### (eerst een weekje spelen en dan) een planning maken
##### vraag om snapshots vd data. vraag ook vanaf welk moment de activiteit terugloopt of: welke gebrukers zijn die lok-gebruikers? (fake-activiteit-vertonende gebruikers)