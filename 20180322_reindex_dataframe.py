# coding: utf-8

# python <file>.py <path-where-input-csvs-are> <path-to-where-output-should-be-saved>

# a successful run will (for 16 csv-files) be finished within a second. A progess bar shows the estimated runtime

# quick copy-paste for me:
# python 20180322_reindex_dataframe.py "C:\Users\sternheimam\Desktop\my-notebook\user-csvs" "C:\Users\sternheimam\Desktop\my-notebook\user-csvs-swirled"
############################################################
## IMPORTS
############################################################
import glob, os, re, sys
import pandas as pd
from tqdm import *

############################################################
## FUNCITON DEFINITIONS
############################################################
def swap_to_front(header, dependent_variable):
	""" this function takes a list of column names, and the column name that is to be placed in front. It places this column name on the [1]th position of the list of column names. """
	header.remove(dependent_variable)
	header.insert(1,dependent_variable)
	return header

def reindex_dataframe(df,dependent_variable):
	""" this function takes a dataframe, and the name of the column that is to be switched to the front. It places this column on the [1]th position in the dataframe. """
	header = df.columns.tolist()
	header = swap_to_front(header,dependent_variable)
	new_df = df.reindex(columns = header)
	return new_df

	
############################################################
## MAIN CODE
############################################################

#---------------------------
# Fetch all .csv-files in specified path, and 'swirl' those around, so you get about a tenfold of the same file, with different columns in front.
#---------------------------
path_in = sys.argv[1] #'C:\Users\sternheimam\Desktop\my-notebook\user-csvs'
path_out = sys.argv[2] #'C:\Users\sternheimam\Desktop\my-notebook\user-csvs-swirled'

for filename in tqdm(glob.glob(os.path.join(path_in, '*.csv'))):
	user = re.search("_(.[0-9]*)\.csv", filename).group(1)
	df = pd.read_csv(filename)
	for column in df:
		if column != 'Date & Time':
			df = reindex_dataframe(df,column)
			column = re.sub('/', '', column) #remove slashes from name (those mess with the file name)
			name = str(user)+"_"+str(column)+".csv"
			df.to_csv(os.path.join(path_out,name),index=False)