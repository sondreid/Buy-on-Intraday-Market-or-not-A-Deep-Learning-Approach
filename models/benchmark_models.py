###########################################################################################################################################
###### ARIMA models #######################################################################################################################
###########################################################################################################################################


# fit an ARIMA model and plot residual errors
from typing import Dict
import pandas as pd
import numpy as np
import math

from pmdarima import auto_arima

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from torch import rand
from utilities import *
import itertools
import time
# Source: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
from statistics import mean
from model_definitions import * 
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)
from random import shuffle


target = 'intraday_price'



series_df_evaluation = remove_nans(series_df_evaluation, 'linear')
no2_intraday_price_evaluation = series_df_evaluation.intraday_price
no2_dayahead_price_evaluation = series_df_evaluation.dayahead_price



arima_orders_dayahead = {

    'p': [i for i in range(2,6)],
    'd': [1],
    'q': [i for i in range(1,8)],
    'P': [0],
    'D': [0],
    'Q': [0],
    's': [0]

}


arima_orders_intraday = {

    'p': [i for i in range(1,6)],
    'd': [1],
    'q': [i for i in range(1,6)],
    'P': [0],
    'D': [0],
    'Q': [0],
    's': [0]

}

arima_orders_intraday_opt = {

    'p': [2],
    'd': [1],
    'q': [2],
    'P': [0],
    'D': [0],
    'Q': [0],
    's': [0]

}




sarima_orders = {

    'p': [i for i in range(2,4)],
    'd': [1],
    'q': [i for i in range(2,4)],
    'P': [i for i in range(1,3)],
    'D': [0, 1],
    'Q': [i for i in range(1,4)],
    's': [24, 12, 24*7]

}


arima_orders_no_season = {

    'p': [i for i in range(1,8)],
    'd': [1],
    'q': [i for i in range(1,8)],
    'P': [0],
    'D': [0],
    'Q': [0],
    's': [0]

}


"""sarima_orders = {

    'P': [i for i in range(2,3)],
    'D': [0],
    'Q': [i for i in range(2,3)],
    'p': [i for i in range(4,5)],
    'd' : [1],
    'q': [i for i in range(4,5)],
    's': [24],

}"""

sarimax_opt_orders = {

    'P': [2],
    'D': [0],
    'Q': [2],
    'p': [4],
    'd' : [1],
    'q': [4],
    's': [24],
    'n':[3],
    'c':[2],
    't':[2],
    'ct':[2]

}


sarimax_orders = {

    'P': [i for i in range(2,3)],
    'D': [0],
    'Q': [i for i in range(2,3)],
    'p': [i for i in range(4,5)],
    'd' : [1],
    'q': [i for i in range(4,5)],
    's': [24],
    'n':[2,3],
    'c':[2,3],
    't':[2,3],
    'ct':[2,3]

}

test_orders =  {
    'P':  2,
    'D':  0,
    'Q':  2,
    'p':  4,
    'd':  1,
    'q':  4,
    's':  24,
    'n':  2,
    'c':  2,
    't':  2,
    'ct': 2
}




ets_orders = {
    'error': ['add'],
    'trend': ['add', None],
    'damped_trend': [True, False],
    'seasonal': ['add', None],
    'seasonal_periods': [12, 24, 48]#, None] # if seasonal is not None
}

ets_config_opt = {
    'error': ['add'],
    'trend': ['add'],
    'damped_trend': [True],
    'seasonal': ['add'],
    'seasonal_periods': [12, 24, 48, 24*7]
}


ets_config = {
    'error': 'add',
    'trend': 'add',
    'damped_trend': True,
    'seasonal': 'add',
    'seasonal_periods': 12
}

def ets_model(train:pd.Series, test:np.array, orders:dict, verbose = False):
    # fit in statsmodels


    if orders.get('trend') == None:
        orders['damped_trend'] = None


    ets = ETSModel(
        endog               = train,
        error               = orders.get('error'), 
        trend               = orders.get('trend'), 
        seasonal            = orders.get('seasonal'), 
        damped_trend        = orders.get('damped_trend'),
        seasonal_periods    = orders.get('seasonal_periods') 
    )
    ets_fit     = ets.fit()
    forecast    = ets_fit.forecast(steps=len(test))
    return forecast, ets_fit



def arima_model(train: pd.Series, test: pd.Series, orders:dict, verbose = False):

    """
    @orders: a single dictionary
    """
    if verbose: 
        print(colored("> Config" + str(orders), "blue"))
    order = tuple([orders[key] for key in ['p', 'd', 'q']])
    seasonal_order = tuple([orders[key] for key in ['P', 'D', 'Q', 's']])
    try:
        trend = tuple([orders[key] for key in ['n', 'c', 't', 'ct']])
        print(" Trend terms", str(trend))
    except:
        trend = None
        print("> Missing trend")
    arima = ARIMA(train, order=order, seasonal_order= seasonal_order, trend = trend, enforce_invertibility=False, enforce_stationarity=False)
    arima_fit = arima.fit()
    forecast = arima_fit.forecast(steps=len(test), alpha = 0.05)
    return forecast, arima_fit


def sarimax_model(train:pd.Series, test:pd.Series, orders, verbose = False):
    
    """
    @orders: a single dictionary
    """
    if verbose: 
        print(colored("> Config" + str(orders), "blue"))
    try:
        seasonal_order = tuple([orders[key] for key in ['P', 'D', 'Q', 's']])
        order = tuple([orders[key] for key in ['p', 'd', 'q']])
        trend = tuple([orders[key] for key in ['n', 'c', 't', 'ct']])
    except:
        raise ValueError("Missing parameters")
    arima = SARIMAX(train, order=order, seasonal_order= seasonal_order, trend = trend, enforce_invertibility=False, enforce_stationarity=False)
    arima_fit = arima.fit()
    forecast = arima_fit.forecast(steps=len(test), alpha = 0.05)
    return forecast, arima_fit


def auto_arima_folds(series: np.array, 
                     cv = cv,
                     verbose = False):
    """
    @function_call: model to fit and predict (memory location)
    """                
    # Initial increment value
    i = 0
    print(colored("> " + cv.get_type() + " cross validation for optimal model pending... ", "green"))

    # Iterate through cross validation nested-list
    for train_indices, val_indices, test_indices in cv.split(series):
        train = series[train_indices]
        print("Length train: ", len(train))
        opt_arima = auto_arima(train, 
                              start_q = 2, 
                              start_p = 2,
                              max_p   = 7,
                              max_q   = 7,
                              start_P = 1,
                              start_Q = 1,
                              max_P   = 2,
                              max_D   = 1,
                              max_Q   = 2,
                              m       = 24)
        print(opt_arima.summary())



def cross_validate(series_df: pd.DataFrame, 
                     orders:tuple,
                     function_call,
                     neptune_logger,
                     cv = cv,
                     verbose = False):
    """
    @function_call: model to fit and predict (memory location)
    """                
    # Initial increment value for print statements
    i = 0
    print(colored("> " + cv.get_type() + " cross validation for optimal model pending... ", "green"))
    

    print("> Target {}".format(target))
    
    # Time the model 
    start_model_time = time.time()
    training_and_val_data = {}
    benchmark_performance = {
        'mean_val_mae'  : [],
        'mean_val_smape' : [],
        'mean_val_mape' : [],
        'mean_val_rmse' : []
    }

    # Iterate through cross validation nested-list
    for train_indices, val_indices, test_indices in cv.split(series_df):

        if verbose: print(colored(">>>>> Fold number: " + str(i + 1) + " of " + str(cv.n_splits), 'green'))
        
        # Console message
        if verbose: print("TRAIN:", min(train_indices), " - " , max(train_indices), '\n', "TEST:", min(test_indices) , " - ", max(test_indices))

        complete_folds = math.floor(len(val_indices)/max_prediction_length)

        # Record performance per fold
        fold_performance = {
            'mean_val_mae'  : [],
            'mean_val_mape' : [],
            'mean_val_smape' : [],
            'mean_val_rmse' : [],
            'mean_val_loss' : []
            }

        for j in range(1, complete_folds+1): 
        

            new_train_indices, new_val_indices, _ = non_overlapping_train_val_folds(j, max_prediction_length, train_indices, val_indices, test_indices)
            
            new_train = remove_nans(series_df.iloc[new_train_indices], 'linear')[target]
            new_val   = remove_nans(series_df.iloc[new_val_indices],   'linear')[target]

            if verbose: 
                print(colored("> Last element in train fold {}".format(new_train_indices[-1])))
                print(colored(">  Val indices {}".format(new_val_indices) ,'cyan'))
            
  
            forecasts,_ = function_call(train = new_train, test = new_val, orders = orders, verbose = verbose)



            actuals_and_fc = pd.DataFrame({'actuals': new_val.to_numpy(), 'forecasts':forecasts.to_numpy()})
            val_metrics = calc_metrics(actuals_and_fc)
            """         val_mae   = mean_absolute_error(new_val, forecasts)
                        val_mape  = mean_absolute_percentage_error(new_val, forecasts)
                        val_smape = smape(new_val, forecasts)
                        val_rmse  = mean_squared_error(new_val, forecasts, squared = False)
            """
            train_make_plot(pd.Series(new_val), pd.Series(forecasts), save ="bench_{}_fold_{}_val_{}".format(str(function_call)[10:], i,j), legend_text= "MAE: {}".format(val_metrics['test_MAE']),  neptune_logger= neptune_logger)

            if verbose: print(colored("> Val MAE {} outer fold number {} inner fold number {}".format(val_mae, i, j)))
            
            # Interim step for total performance calculation 
            benchmark_performance['mean_val_mae'].append(val_metrics['test_MAE'])
            benchmark_performance['mean_val_mape'].append(val_metrics['test_MAPE'])
            benchmark_performance['mean_val_smape'].append(val_metrics['test_SMAPE'])
            benchmark_performance['mean_val_rmse'].append(val_metrics['test_RMSE'])

            fold_performance['mean_val_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            fold_performance['mean_val_mape'].append(val_metrics['test_MAPE'])
            fold_performance['mean_val_smape'].append(val_metrics['test_SMAPE'])
            fold_performance['mean_val_rmse'].append(val_metrics['test_RMSE'])

            print(colored(">>> End of walkforward prediction {} of {} val metrics {} ".format(j, complete_folds, val_metrics['test_MAE']), "green"))
            print(colored(">>> End of walkforward averge val mae so far {} ".format(mean(benchmark_performance['mean_val_mae'])), "green"))
            #### END OF WALKFORWARD PREDICTION ###

        

        i += 1  # Index increment 


        new_train_val_data_dict = {
                        "train_indices": new_train_indices, 
                        "val_indices": new_val_indices, 
                        'val_mae'    : mean(fold_performance['mean_val_mae']), 
                        'val_rmse'   : mean(fold_performance['mean_val_rmse']), 
                        'val_mape'   : mean(fold_performance['mean_val_mape']), 
                        'val_smape'  : mean(fold_performance['mean_val_smape'])}
    
        training_and_val_data[i] = new_train_val_data_dict 

    if neptune_logger is not None:
        neptune_logger['model/training_and_val_data'] = training_and_val_data
    # Calculate standard deviation
    benchmark_performance['sd_val_mae']   =  np.std(benchmark_performance['mean_val_mae'])
    benchmark_performance['sd_val_smape'] =  np.std(benchmark_performance['mean_val_smape'])
    benchmark_performance['sd_val_mape']  =  np.std(benchmark_performance['mean_val_mape'])
    benchmark_performance['sd_val_rmse']  =  np.std(benchmark_performance['mean_val_rmse'])


    benchmark_performance['mean_val_mae'] = mean(benchmark_performance['mean_val_mae'])
    benchmark_performance['mean_val_smape'] = mean(benchmark_performance['mean_val_smape'])
    benchmark_performance['mean_val_mape'] = mean(benchmark_performance['mean_val_mape'])
    benchmark_performance['mean_val_rmse'] = mean(benchmark_performance['mean_val_rmse'])


    # Calculate time spent on model training and forecasting in seconds
    end_model_time = time.time()
    time_used = end_model_time - start_model_time

    print(colored("> Model orders [ (pdqPDQ)[m] if ARIMA ] :" + str(orders) + " <-> Cross-validated avg test MAE:" + str(round(benchmark_performance['mean_val_mae'] , 2)) + " <-> Time used:" + str(round(time_used)) + " seconds", "blue"))

    return {"Model (pdqPDQm)": orders, "performance": benchmark_performance}


# Uncomment to test

"""

arima_orders_dayahead = {

    'p': 2,
    'd': 1,
    'q': 2,
    'P': 0,
    'D': 0,
    'Q': 0,
    's': 0

}
target = 'intraday_price'
performance_1 = cross_validate(series_df, arima_orders_dayahead, arima_model, None, cv, verbose = True )



arima_orders_dayahead = {

    'p': 3,
    'd': 0,
    'q': 4,
    'P': 0,
    'D': 0,
    'Q': 0,
    's': 0

}



performance_2 = cross_validate(series_df, arima_orders_dayahead, arima_model, None, cv )

# refit and test

test = no2_intraday_price.iloc[5000:5039]

#test = no2_intraday_price_evaluation.iloc[0:40]

forecasts = ets_model(train = no2_intraday_price.iloc[0:5000], test = test, orders = {'error': 'add', 'trend': 'add', 'damped_trend': True, 'seasonal': 'add', 'seasonal_periods': 24})

forecasts  = sarimax_model(train = no2_intraday_price, test = no2_intraday_price_evaluation.iloc[0:40], orders = test_orders)

forecasts_and_actuals = pd.DataFrame({'seq' : np.arange(0, len(forecasts)).tolist(), 'forecasts': forecasts, 'actuals':test})


forecasts_and_actuals_long = forecasts_and_actuals.melt(id_vars = ['seq'], 
                var_name = 'type', 
                value_name = 'euro_mwh', value_vars = ['forecasts', 'actuals'])

sns_lineplot(df = forecasts_and_actuals_long, x = forecasts_and_actuals_long.seq, y = forecasts_and_actuals_long.euro_mwh, hue = 'type' )




mean_squared_error(forecasts_and_actuals.actuals, forecasts_and_actuals.forecasts)


"""


def grid_search(df: pd.DataFrame, function_call, orders:Dict[str, list], filename:str):

    """
    Wrapper function for perform_grid_search arima

    @orders: dictionary of keys and lists of possible orders
    @function_call: model to fit and predict (memory location)
    """
    print(colored("> Starting grid search with following orders" + str(orders), "green"))
    keys, values = zip(*orders.items())
    all_orders_list = [dict(zip(keys, v)) for v in itertools.product(*values)] 
    print(colored("All orders {}".format(all_orders_list), 'green'))
    __perform_grid_search(df, function_call, all_orders_list, filename=filename)


def __perform_grid_search(df, function_call, all_orders_list:list, filename:str):
    """
    Private method
    Performs grid search for ARIMA
    @function_call: model to fit and predict (memory location)
    """
    
    model_performance = pd.DataFrame()    
    j = 0

    best_model, best_score, best_metrics = None, np.inf, np.inf

    neptune_logger = neptune.init(
        project         = "MasterThesis/Price",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        name = str(function_call)[10:],
        description = str(function_call)[10:], 
        source_files=["benchmark_models.py", "model_definitions.py"]
    )  
    neptune_logger['Type']                                              = "Cross validated run"
    neptune_logger['Target']                                            = target
    neptune_logger['Architecture']                                      = str(function_call)[10:]
    

    if not os.path.exists("manual_logs"):
        os.makedirs("manual_logs")
    if len(all_orders_list) > 1:
        for orders in all_orders_list:
                
            j += 1
            
            print('> ' + str(j) + ' out of ' + str(len(all_orders_list)))
            print('> Order in trial: ' + str(orders))

            try:
                model_performance = cross_validate(df, orders, function_call, neptune_logger = None)
                print("> Model performance {}".format(model_performance.get('performance').get('mean_val_mae') ))
                if model_performance.get('performance').get('mean_val_mae') < best_score:
                    best_model  = model_performance.get('Model (pdqPDQm)')
                    best_score  = model_performance.get('performance').get('mean_val_mae')
                    best_orders  = orders
                    best_metrics = model_performance.get('performance')
            except:
                raise ValueError("> Invalid configuration")
        print(colored(" > Running new run with optimal orders"))
        model_performance = cross_validate(df, best_orders, function_call, neptune_logger = neptune_logger)
    else:
        model_performance = cross_validate(df, all_orders_list[0], function_call, neptune_logger = neptune_logger)
        best_model  = model_performance.get('Model (pdqPDQm)')
        best_score  = model_performance.get('performance').get('mean_val_mae')
        best_metrics = model_performance.get('performance')
    neptune_logger['model/total_performance']      = best_metrics
    neptune_logger['model/configuration']          = best_model
    neptune_logger.stop()
    with open('manual_logs/' + 'benchmark/' + filename + '.txt', 'w') as f:
        f.writelines('Best model: Model orders :' + str(best_model))
        f.writelines('\n')
        f.writelines('Metrics : {}'.format(best_metrics))



    print("> Best model: " + str(best_model) + "-> Best score" + str(best_score))
    return best_model, best_score


def random_search(df: pd.DataFrame, function_call,  orders:Dict[str, list], n:int, filename:str):
    """
    Performs random search by shuffling a dictionary of possible combinations, and selecting 
    n first combinations
    @orders: dictionary of keys and lists of possible orders
    @function_call: model to fit and predict (memory location)
    """
    print(colored("> Starting random search with following orders" + str(orders), "green"))
    keys, values = zip(*orders.items())
    all_orders_list = [dict(zip(keys, v)) for v in itertools.product(*values)] 
    shuffle(all_orders_list) # Shuffle and pick the n first combinations
    return __perform_grid_search(df, function_call, all_orders_list[0:n], filename=filename)




if __name__ == '__main__':
    
    target = 'intraday_price'
    grid_search(series_df_shifted, ets_model, orders = ets_config_opt, filename='intraday_ets')
    
    target = 'dayahead_price'
    grid_search(series_df_shifted, ets_model, orders = ets_config_opt, filename='dayahead_ets')
    
    


    #target = 'dayahead_price'
    #grid_search(series_df, ets_model, orders = ets_config_opt, filename='intraday_ets')
    #random_search(series_df, arima_model, orders = sarima_orders, n=5, filename='dayahead_sarima')
    
    #grid_search(no2_dayahead_price, ets_model, orders = ets_orders, filename='dayahead_ets2')
    #grid_search(series_df, arima_model, orders = arima_orders_intraday_opt, filename='dayahead_arima')
    #grid_search(no2_dayahead_price, arima_model, orders = arima_orders_intraday, filename='dayahead_arima_2')
    #random_search(series_df, arima_model, orders = arima_orders_intraday, n=30, filename='intraday_arima')
    #target = 'dayahead_price'
    #random_search(series_df, sarimax_model, orders = sarima_orders, n=20, filename='dayahead_sarima')
    #random_search(no2_dayahead_price, arima_model, orders = sarima_orders, n=20, filename='dayahead_sarima')
    #random_search(no2_dayahead_price, arima_model, orders = arima_orders_intraday, n=20, filename='dayahead_arima')
    #random_search(no2_intraday_price, sarimax_model, sarimax_orders, n=5, filename='intraday_sarimax1')
    #random_search(no2_intraday_price, sarimax_model, sarimax_orders, n=20, filename='intraday_sarimax2')
    #random_search(no2_dayahead_price, sarimax_model, sarimax_orders, n=20, filename='dayahead_sarimax2')
    

