import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import datetime
from datetime import timedelta
import time
import sys
import helpers as hp

tic = time.time()

f_cfg = sys.argv[1]
f_out = sys.argv[2]

(sample_size, stock_mean, stock_stdev, stock_norm_factor, 
	no_stock, stock_sample_size, vol_lb, vol_ub, corr_alpha, 
	corr_beta, no_mat, mat_lb, mat_ub, mat_sample_size, 
	stock_simulation_shape, no_corr, corr_simulation_shape, 
	mat_simulation_shape, val_date, settle_date, rf_rate, 
	dividend, strike) = hp.loadVariables(f_cfg)


stock_fwd_prices = stock_norm_factor \
					* np.random.lognormal(stock_mean, 
											stock_stdev, 
											stock_simulation_shape)

stock_vol = np.random.uniform(vol_lb, vol_ub, stock_simulation_shape)

stock_corr = np.random.beta(corr_alpha, corr_beta, corr_simulation_shape)

maturity = np.random.uniform(mat_lb,mat_ub,mat_simulation_shape)**2


df = hp.loadDataInDataFrame(sample_size, val_date, settle_date, maturity, 
						rf_rate, strike, dividend, no_stock, stock_fwd_prices, 
						stock_vol, stock_corr, no_corr)

df['price'] = df.apply(lambda row: hp.getPriceFromRow(row),axis=1)

df.to_csv(f_out, index=False)

toc = time.time()
print('Total runtime is %d seconds' % (toc-tic))

print('--done!')