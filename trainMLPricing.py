from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import time
import sys

f_sample = sys.argv[1] # "d:/mark_at_home/playground/test1.csv"
f_out_dir = sys.argv[2]

df_sample = pd.read_csv(f_sample)

X_sample = df_sample[['days_to_maturity','stock_0','strike','vol_0','dividend','rf_rate']].values
Y_sample = df_sample[['price']].values

no_hidden_layer = 6
no_neuron_list = [10, 1200, 1400, 1600] 
sample_size = 1000000
X_Lite = X_sample[:sample_size]
Y_Lite = Y_sample[:sample_size]

X_train, X_test, y_train, y_test = train_test_split(X_Lite, Y_Lite, random_state = 0)
_activation = 'relu'
#_alpha = 0.0001
#_hidden_layer_size = [100,100,100,100,100,100]
_solver = 'adam'

for this_no_neuron in no_neuron_list:
    print("Running model with %d hidden layers, and %d neurons" % (no_hidden_layer, this_no_neuron))
    _hidden_layer_size = [this_no_neuron] * no_hidden_layer
    tic = time.time()
    mlpreg = MLPRegressor(hidden_layer_sizes = _hidden_layer_size,
                                 activation = _activation,
                                 #alpha = _alpha,
                                 solver = _solver).fit(X_train, y_train)
    toc = time.time()
    print("Run time is %f " % (toc - tic))
    print("score on training set is %f" % (mlpreg.score(X_train, y_train)))
    print("score on test set is %f" % (mlpreg.score(X_test, y_test)))
    out_f = f_out_dir + '/ML_Pricing_Model_1MM_adam_%d.sav' % this_no_neuron
    pickle.dump(mlpreg, open(out_f, 'wb'))

