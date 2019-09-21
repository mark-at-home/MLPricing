import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import datetime
from datetime import timedelta
import time
import json

def loadCfg(f_cfg):
	with open(f_cfg, 'r') as f:
		dict_cfg = json.load(f)
	return dict_cfg

def loadVariables(f_cfg):

	dict_cfg = loadCfg(f_cfg)

	sample_size = int(dict_cfg['sample_size'])
	# stock parameters (lognormal)
	stock_mean = float(dict_cfg['stock_mean'])
	stock_stdev = float(dict_cfg['stock_stdev'])
	stock_norm_factor = float(dict_cfg['stock_norm_factor'])
	no_stock = int(dict_cfg['no_stock'])
	stock_sample_size = sample_size

	# stock vol parameters (uniform)
	vol_lb = float(dict_cfg['vol_lb'])
	vol_ub = float(dict_cfg['vol_ub'])

	# stock correlation parameters (beta)
	corr_alpha = float(dict_cfg['corr_alpha'])
	corr_beta = float(dict_cfg['corr_beta'])

	# option maturity parameters (uniform)
	no_mat = int(dict_cfg['no_mat']) # per option
	mat_lb = float(dict_cfg['mat_lb'])
	mat_ub = float(dict_cfg['mat_ub'])
	mat_sample_size = sample_size


	# simulation shapes
	stock_simulation_shape = (stock_sample_size, no_stock)
	no_corr = int(no_stock * (no_stock - 1) /2)
	corr_simulation_shape = (stock_sample_size, no_corr)
	mat_simulation_shape = (mat_sample_size, no_mat)

	# Valuation CFG
	val_date = dict_cfg['val_date']
	settle_date = dict_cfg['settle_date']
	rf_rate = float(dict_cfg['rf_rate'])
	dividend = float(dict_cfg['dividend'])
	strike = float(dict_cfg['strike'])

	outSet = (sample_size, stock_mean, stock_stdev, stock_norm_factor, 
				no_stock, stock_sample_size, vol_lb, vol_ub, corr_alpha, 
				corr_beta, no_mat, mat_lb, mat_ub, mat_sample_size, 
				stock_simulation_shape, no_corr, corr_simulation_shape, 
				mat_simulation_shape, val_date, settle_date, rf_rate, 
				dividend, strike)

	return outSet

def populateCorrMatrix(corr_vector, no_stock):
	corr_update = corr_vector.copy()
	matrix = ql.Matrix(no_stock,no_stock)
	for i in range(no_stock):
		for j in range(no_stock):
			if i == j:
				matrix[i][j] = 1.0
				# print("%d,%d,1.0" %(i,j))
			elif i > j:
				matrix[i][j] = corr_update[0]
				corr_update = corr_update[1:]
				# print("%d,%d,%f" %(i,j, matrix[i][j]))
	for i in range(no_stock):
		for j in range(no_stock):
			if i < j:
				matrix[i][j] = matrix[j][i]
				# print("%d,%d,%f" %(i,j, matrix[i][j]))\
	return matrix

def valueBasketCall(val_date,
					settle_date,
					maturity_days,
					tmp_stock,
					tmp_vol,
					tmp_corr,
					rf_rate=0.0,
					strike=100.0,
					dividend=0.0,
					basketCallType='Min',
					randMethod="pseudorandom",
					_timeStepsPerYear=1,
					_requiredTolerance=0.02,
					_seed=42):
	
	tic = time.time()
	
	# Input Checks
	no_stock = len(tmp_stock)
	theo_no_corr = no_stock * (no_stock - 1) /2
	
	if len(tmp_corr) != theo_no_corr:
		print('Number of stocks (%d) and number of correlations (%d) do not match! Should be %d ' % (no_stock, len(tmp_corr), theo_no_corr))
		return 0.0
	
	if basketCallType not in ["Min","Max","Average"]:
		print('Unsupported basket Call type %s, only ["Min","Max","Average"] are supported.' % basketCallType)
		return 0.0
	
	# Set dates
	day_count = ql.Actual365Fixed()
	py_val_date = datetime.strptime(val_date, '%Y-%m-%d')
	ql_val_date = ql.Date(py_val_date.day, py_val_date.month, py_val_date.year)

	py_settle_date = datetime.strptime(settle_date, '%Y-%m-%d')
	ql_settle_date = ql.Date(py_settle_date.day, py_settle_date.month, py_settle_date.year)
	
	py_maturity_date = py_val_date + timedelta(days=int(maturity_days))
	ql_maturity_date = ql.Date(py_maturity_date.day, py_maturity_date.month, py_maturity_date.year)
	
	# Set QL instance
	ql.Settings.instance().evaluationDate = ql_val_date

	# Create exercise object
	exercise = ql.EuropeanExercise(ql_maturity_date)
	
	# Create Payoff Object
	payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
	


	# Create risk free rate object
	riskFreeRate = ql.FlatForward(ql_settle_date, rf_rate, day_count)


	
	# Create list of equity parameters
	ql_underlying_list = [ql.SimpleQuote(i) for i in tmp_stock]
	ql_vol_list = [ql.BlackConstantVol(ql_val_date, ql.TARGET(), i, day_count) for i in tmp_vol]
	ql_div_list = [ql.FlatForward(ql_settle_date, dividend, day_count) for i in tmp_stock]

	# Populate Correlation Matrix
	matrix = populateCorrMatrix(tmp_corr, no_stock)
	
	# Create list of QL processes for equity
	ql_process_list =[ql.BlackScholesMertonProcess(    
		ql.QuoteHandle(ql_underlying_list[i]),
		ql.YieldTermStructureHandle(ql_div_list[i]),
		ql.YieldTermStructureHandle(riskFreeRate),
		ql.BlackVolTermStructureHandle(ql_vol_list[i]),) for i in range(len(ql_underlying_list))]
	
	if no_stock == 1:
		# Use Black Scholes Formula to calculate European Call
		#print("Single stock, use BS Formula!")
		option = ql.VanillaOption(payoff, exercise)
		bsm_process = ql_process_list[0]
		option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
	else:
		# Create QL Multi-variate Stochastic Process
		process = ql.StochasticProcessArray(ql_process_list, matrix)
		
		# Create Basket Option Object
		if basketCallType == "Min":
			option = ql.BasketOption(ql.MinBasketPayoff(payoff), exercise)
		elif basketCallType == "Max":
			option = ql.BasketOption(ql.MaxBasketPayoff(payoff), exercise)
		elif basketCallType == "Average":
			option = ql.BasketOption(ql.AverageBasketPayoff(payoff), exercise)

		# Set Pricing Engine
		option.setPricingEngine(ql.MCEuropeanBasketEngine(process, 
			  	randMethod, 
			  	timeStepsPerYear=_timeStepsPerYear, 
			  	requiredTolerance=_requiredTolerance, 
			  	seed=_seed)
		)
		
	p = option.NPV()
	#p=0
	toc = time.time()
	
	#print(p, maturity_days, toc-tic)
	return p


def getPriceFromRow(row):
	
	stock_cols = [x for x in row.index.tolist() if x.startswith('stock_')]
	vol_cols = [x for x in row.index.tolist() if x.startswith('vol_')]
	corr_cols = [x for x in row.index.tolist() if x.startswith('corr_')]

	tmp_stock = np.array([row[x] for x in stock_cols])
	tmp_vol = np.array([row[x] for x in vol_cols])
	tmp_corr = np.array([row[x] for x in corr_cols])
	
	#tic = time.time()
	p = valueBasketCall(row['datum'],
					row['settle_date'],
					row['days_to_maturity'],
					tmp_stock,
					tmp_vol,
					tmp_corr,
					rf_rate=float(row['rf_rate']),
					strike=float(row['strike']),
					dividend=float(row['dividend']),
					basketCallType='Min',
					randMethod="pseudorandom",
					_timeStepsPerYear=1,
					_requiredTolerance=0.02,
					_seed=42)
	#toc = time.time()
	
	#print(p,toc-tic)
	return p


def loadDataInDataFrame(sample_size, val_date, settle_date, maturity, 
						rf_rate, strike, dividend, no_stock, stock_fwd_prices, 
						stock_vol, stock_corr, no_corr):
	data = {
		"datum":[val_date]*sample_size,
		"settle_date":[settle_date]*sample_size,
		"days_to_maturity":maturity[:,0],
		"rf_rate":[rf_rate]*sample_size,
		"strike":[strike]*sample_size,
		"dividend":[dividend]*sample_size
	}

	data_stock = {}
	data_stock_vol = {}
	data_stock_corr = {}
	for i in range(no_stock):
		stock_col_id = 'stock_%d' % i
		vol_col_id = 'vol_%d' % i
		data_stock[stock_col_id] = stock_fwd_prices[:,i]
		data_stock_vol[vol_col_id] = stock_vol[:,i]

	for i in range(no_corr):
		corr_col_id = 'corr_%d' % i
		data_stock_corr[corr_col_id] = stock_corr[:,1]
		
	data.update(data_stock)
	data.update(data_stock_vol)
	data.update(data_stock_corr)
	#print(np.shape([val_date]*sample_size))
	#print(np.shape([val_date]*sample_size))
	#print(np.shape([settle_date]*sample_size))
	#print(np.shape(maturity[:,0]))
	#print(np.shape(stock_fwd_prices))

	df = pd.DataFrame(data)

	return df