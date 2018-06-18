import os # for path
import pandas as pd # for dataframes
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
	
path_in = "C:\Users\sternheimam\Desktop\my-notebook\user-csvs_predictions123"
# get the data from the files
data = Get_data(path_in) # all files

print sum(data[0]),sum(data[1]),sum(data[2])