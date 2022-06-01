"""

Non-fitted benchmark models which relies on simple statistics, or
pre-calculated models (EQ benchmarks)

"""

from model_definitions import * 
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from models.utilities import remove_nans



# Loading EQ forecasts

no2_dayahead_eq_forecasts = pd.read_csv("../model_data/eq_dayahead_forecasts.csv")
no2_dayahead_eq_forecasts.datetime = pd.to_datetime(no2_dayahead_eq_forecasts.datetime, format='%Y-%m-%d %H:%M:%S')

no2_dayahead_eq_short_term_forecasts = pd.read_csv("../model_data/eq_short_term_dayahead_forecasts.csv")
no2_dayahead_eq_short_term_forecasts.datetime = pd.to_datetime(no2_dayahead_eq_short_term_forecasts.datetime, format='%Y-%m-%d %H:%M:%S')



def take_last_benchmark(df: pd.DataFrame,  prediction_length : int, target : str) -> pd.Series:
    """
    Benchmark model:Last observation in series
    """
    last_observation = df[target].iloc[-1]
    return pd.Series([last_observation for x in range(0, prediction_length)])




def mean_benchmark(df: pd.DataFrame,  prediction_length : int, target : str) -> pd.Series:
    """
    Mean of allf
    """
    mean_series = mean(df[target])
    return pd.Series([mean_series for x in range(0, prediction_length)])




def eq_dayahead_benchmark(df: pd.DataFrame , prediction_length :int, target : str) -> pd.Series:
    """
    Energy quantified f
    """
    try: 
        start_date = pd.to_datetime(df.datetime.iloc[-1],  format='%Y-%m-%d %H:%M:%S')
        utc_dates = start_date.tz_localize('UTC')

        cet_date = utc_dates.tz_convert('Europe/Berlin') + timedelta(hours = -1)
    except:
        cet_date = pd.to_datetime(df.datetime.iloc[-1],  format='%Y-%m-%d %H:%M:%S')

    forecasts = no2_dayahead_eq_forecasts[no2_dayahead_eq_forecasts['datetime'] > cet_date]

    return (forecasts.iloc[0:prediction_length])['NO2 Price Spot EUR/MWh H Forecast : prefix']



def eq_short_term_dayahead_benchmark(df: pd.DataFrame , prediction_length :int, target :str ) -> pd.Series:
    """
    Energy quantified f
    """
    try: 
        start_date = pd.to_datetime(df.datetime.iloc[-1],  format='%Y-%m-%d %H:%M:%S')
        utc_dates = start_date.tz_localize('UTC')

        cet_date = utc_dates.tz_convert('Europe/Berlin') + timedelta(hours = -1)
    except:
        cet_date = pd.to_datetime(df.datetime.iloc[-1],  format='%Y-%m-%d %H:%M:%S')

    forecasts = no2_dayahead_eq_short_term_forecasts[no2_dayahead_eq_short_term_forecasts['datetime'] > cet_date]

    return (forecasts.iloc[0:prediction_length])['NO2 Price Spot Short-term EUR/MWh H Forecast : ']

"""evaluation = series_df_evaluation.iloc[0:40]


eq_short_term_dayahead_benchmark(series_df, 40)
eq_dayahead_benchmark(series_df, 40)
#no2_dayahead_eq_forecasts[no2_dayahead_eq_forecasts['datetime'] >  pd.to_datetime(series_df.datetime.iloc[-1],  format='%Y-%m-%d %H:%M:%S')]



preds_and_actuals = pd.DataFrame({'seq':np.arange(0, len(evaluation.dayahead_price)).tolist(), 'datetime': evaluation.datetime, 'actuals': evaluation.dayahead_price.tolist(), 'preds':eq_dayahead_benchmark(series_df, 40).tolist()})

plot_predictions(preds_and_actuals, "test")


smape(eq_dayahead_benchmark(series_df, 40),evaluation.dayahead_price)

mean_absolute_percentage_error(eq_dayahead_benchmark(series_df, 40).to_numpy(),evaluation.dayahead_price.to_numpy())
"""



def cross_validation_benchmarks(df: pd.DataFrame, 
                     benchmark_model,    
                     cv,
                     model_name:str,    
                     target : str,    
                     verbose = False) -> tuple:
    """
    Function that cross validates a given model configuration and returns the mean of a set of performance
    metrics
    """


    neptune_logger = neptune.init(
            project         = "MasterThesis/Price",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            name = model_name,
            description = model_name,
            source_files=["model_definitions.py", "run_lstm.py", "run_tft.py", "run_gru.py"]
        )  


    neptune_logger['Type']                                  = "Benchmark"
    neptune_logger['Target']                                = target
    neptune_logger['Architecture']                          = model_name


    
    # Initial values and dictionary structures
    total_performance = {
        'mean_val_mae'  : [],
        'mean_val_mape' : [],
        'mean_val_smape' : [],
        'mean_val_rmse' : []
    }
    training_and_val_data    = {}



    # Initial increment value
    i = 0
    print(colored(">" + cv.get_type() + " cross validation pending... ", "green"),  colored("[", "green"), colored(i, "green"), colored("out of", "green"), colored(cv.get_n_splits(), "green"), colored("]", "green"))
    # Iterate through cross validation nested-list
    
    for train_indices, val_indices, test_indices in cv.split(df):


        print(colored(">>>>> Fold number:" + str(i), 'green'))
        
        complete_folds =  math.floor(len(val_indices)/max_prediction_length) # Complete validation folds
        print(colored("> Starting walk-forward forecasts on validation set", "cyan"))
        for j in range(1, complete_folds+1):
            

            new_train_indices, new_val_indices, _ = non_overlapping_train_val_folds(j, max_prediction_length, train_indices, val_indices, test_indices)
            
            new_train =  remove_nans(df, 'linear').iloc[new_train_indices]
            new_val =    remove_nans(df, 'linear').iloc[new_val_indices]
            
            if verbose: 
                print(colored("> Last element in train fold {}".format(new_train_indices[-1])))
                print(colored(">  Val indices {}".format(new_val_indices) ,'cyan'))

            forecasts = benchmark_model(new_train, len(new_val), target) # To ensure val and prediction length never differ

            val_mae   = mean_absolute_error(new_val[target], forecasts)
            val_mape  = mean_absolute_percentage_error(new_val[target], forecasts)
            val_smape = smape(new_val[target], forecasts)
            val_rmse  = mean_squared_error(new_val[target], forecasts, squared = False)

            
            print(colored(" Val MAE {}".format(val_mae), "blue"))
        

            
            train_make_plot(new_val[target], forecasts, save ="_{}_fold_{}_val_{}".format(model_name, i,j), legend_text= "MAE: {}".format(val_mae),  neptune_logger=neptune_logger)
    

            # Interim step for total performance calculation 
            total_performance['mean_val_mae'].append(val_mae)
            total_performance['mean_val_rmse'].append(val_rmse)
            total_performance['mean_val_mape'].append(val_mape)
            total_performance['mean_val_smape'].append(val_smape)

            print(colored(">>> End of walkforward prediction {} of {} val metrics {} ".format(j, complete_folds, val_mae), "green"))
            print(colored(">>> End of walkforward averge val mae so far {} ".format(mean(total_performance['mean_val_mae'])), "green"))
            #### END OF WALKFORWARD PREDICTION ###



        i += 1  # Index increment

        new_train_val_data_dict = {"train_indices": new_train_indices, 
                        "val_indices": new_val_indices, 
                        'val_MAE'    : mean(total_performance['mean_val_mae']), 
                        'val_RMSE'   :mean(total_performance['mean_val_rmse']), 
                        'val_MAPE'   : mean(total_performance['mean_val_mape']), 
                        'val_SMAPE'  : mean(total_performance['mean_val_smape'])}
    
        training_and_val_data[i] = new_train_val_data_dict 

    
    # Calculate standard deviation
    total_performance['sd_val_mae']   =  np.std(total_performance['mean_val_mae'])
    total_performance['sd_val_smape'] =  np.std(total_performance['mean_val_smape'])
    total_performance['sd_val_mape']  =  np.std(total_performance['mean_val_mape'])
    total_performance['sd_val_rmse']  =  np.std(total_performance['mean_val_rmse'])



    #  Performance result :Calulate mean of all fold performance
    total_performance['mean_val_mae']    = mean(total_performance['mean_val_mae'] )
    total_performance['mean_val_mape']   = mean(total_performance['mean_val_mape'] )
    total_performance['mean_val_smape']  = mean(total_performance['mean_val_smape'] )
    total_performance['mean_val_rmse']   = mean(total_performance['mean_val_rmse'] )



    
    # Console message and results
    print(colored("\n"+ ">"+ cv.get_type()+ "cross Validation complete...", "blue"))
    print(colored(">>> Mean Performance Result:", "green"), colored(total_performance, "green"), colored("<<<", "green"))

    neptune_logger['model/total_performance']     = total_performance
    neptune_logger['model/training_and_val_data'] = training_and_val_data
    neptune_logger.stop()

    manual_test_logs({'total_performance':total_performance, 'training and validation data': training_and_val_data}, model_name)
    return total_performance, training_and_val_data


if __name__ == '__main__':

    series_df = remove_nans(series_df_shifted, 'linear')
    target = 'dayahead_price'
    cross_validation_benchmarks(series_df, eq_short_term_dayahead_benchmark, cv, "EQ  short term dayahead benchmark", target,  verbose= True)
