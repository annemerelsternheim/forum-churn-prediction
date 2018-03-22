from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sys



def time_shift_data(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg
## name and main variables are set once the file is run as 'main' function (e.g. in command line)
## the main check ensures that you can only execute the code as main function, not when called from another function.
if __name__=="__main__":

        # python lstm_series.py <csv> <categorical features (string, quoted and ';' separated)> <number of hours used for prediction> <number of iterations (likely values: 50, 100, 300)>
        
        # Use a header with labels for the variables
        # first value: time stamp; second value: dependent variable; rest: independent variables
        # :,: separated
        # e.g.

        # date,pollution,dew,temp,press,wnd_dir,wnd_spd,snow,rain
        # 2010-01-02 00:00:00,129.0,-16,-4.0,1020.0,SE,1.79,0,0
        # 2010-01-02 01:00:00,148.0,-15,-4.0,1020.0,SE,2.68,0,0

        # Use dense times: zero values for absent observations, and "_" for categorical variables


#

        #  E.g.: python lstm_timeseries.py pollution.csv "4" 1
        ## sys.argv[] lets you specify variables on the command line
        dataset = read_csv(sys.argv[1], header=0, index_col=0)
        
        values = dataset.values
        
        categoricalFeatA=[int(x) for x in sys.argv[2].split(";")] # "3;5" => 0 based

        for n in categoricalFeatA:
                encoder = LabelEncoder()
                values[:,n] = encoder.fit_transform(values[:,n])

        values = values.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # number of lag hours (=past N hours)
        n_hours = int(sys.argv[3])
        n_features = len(dataset.columns)
        last_feats=n_features-1

        shifted = time_shift_data(scaled, n_hours, 1)
        
        values = shifted.values
        n_train_hours = 365 * 24
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        n_obs = n_hours * n_features
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]

        # reshape for LSTM
        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

        # Plot training/test error
        
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
 
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
        inv_yhat = concatenate((yhat, test_X[:, -last_feats:]), axis=1)

        # revert back to original non-scaled data
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        # calculate and print RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
