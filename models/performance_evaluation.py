
#!/usr/bin/env python3
#shebang

from model_definitions import  *
from utilities import * 
from run_lstm import * 
from run_deepar import *
from benchmark_models import * 
from benchmarks import *

prediction_length  = 38

cv = get_cv_type('sliding_evaluation', 5, **{'val_size':114})




def evaluation_pytorch_models(df, model, model_configuration, timeseries_kwargs,  train_indices,  val_indices, test_indices, **kwargs) -> pd.DataFrame:

    

        """
        sumary_line
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        
        
        data_module = DataModuleTimeSeries(df = df, train_indices = train_indices, val_indices=val_indices, test_indices=val_indices + test_indices, **timeseries_kwargs)
        
        trainer, model = train_model(data_module=data_module, model= model, neptune_logger = None, gpu_bool=gpus,model_configuration=model_configuration, **kwargs)
                       
        
        try: 
            model_call = model_configuration.get("model_call")
            
        except:
                raise ValueError("Missing stored")
        

        # Perform ensembles, load from best checkpoint, or take last epoch 
        if model_configuration.get("k_best_checkpoints") is not None:
            print(colored("> Model call {}".format(model_call), 'green'))
            print(colored(" > Using best k best", 'green'))
            actuals_and_fc = ensemble_checkpoints(trainer, model_call, data_module.test_dataloader())

        elif model_configuration.get('use_best_checkpoint'): 

            print(colored("> Using best checkpoint....", 'green'))
            best_model_path = trainer.checkpoint_callback.best_model_path
            model = model_call.load_from_checkpoint(best_model_path)
            actuals_and_fc = perform_predictions(model, data_module.test_dataloader())
        else:

            actuals_and_fc = perform_predictions(model, data_module.test_dataloader())
        train_logged_metrics = dict([ (k,r.item()) for k,r in trainer.logged_metrics.items()]) 
        actuals_and_fc['datetime'] = df.iloc[test_indices]['datetime'].tolist()
        return actuals_and_fc, train_logged_metrics


def evaluation_benchmarks(df, model, model_configuration, timeseries_kwargs,  train_indices,  val_indices, test_indices, **kwargs):
    
    """
    Running non-fitted benchmarks
    """
    
    target = timeseries_kwargs.get('target')

    new_train = remove_nans(series_df_shifted.iloc[train_indices], 'linear')
    new_test   = remove_nans(series_df_shifted.iloc[test_indices],   'linear')

    forecasts = model(new_train, len(new_test), target)

    print(" > Target {}".format(target))

    actuals_and_fc = pd.DataFrame({'actuals':new_test[target].to_numpy(), 'forecasts':forecasts.to_numpy(), 'datetime': new_test.datetime.to_numpy()})

    train_logged_metrics = {'train_loss_epoch':       0,
                            'train_MAE_epoch':        0,
                            'train_SMAPE_epoch':      0,
                            'train_RMSE_epoch':       0 }

    return actuals_and_fc, train_logged_metrics


def evaluation_benchmark_models(df, model, model_configuration, timeseries_kwargs,  train_indices,  val_indices, test_indices, **kwargs):
    """
    Non-neural model architectures
    """

    target = timeseries_kwargs.get('target')

    new_train = remove_nans(series_df_shifted.iloc[train_indices], 'linear')[target]
    new_test   = remove_nans(series_df_shifted.iloc[test_indices],   'linear')


    forecasts, fit = model(train = new_train, test = new_test[target].tolist(), orders = model_configuration, verbose = False)
    
    print(" > Target {}".format(target))

    train_logged_metrics = {'train_loss_epoch':      mean_absolute_error(fit.fittedvalues, new_train),
                            'train_MAE_epoch':       mean_absolute_error(fit.fittedvalues, new_train),
                            'train_SMAPE_epoch':     smape(fit.fittedvalues, new_train),
                            'train_RMSE_epoch':      mean_squared_error(fit.fittedvalues, new_train, squared = False) }


    actuals_and_fc = pd.DataFrame({'actuals': new_test[target].to_numpy(), 'forecasts':forecasts.to_numpy(), 'datetime': new_test.datetime.to_numpy()})

    return actuals_and_fc, train_logged_metrics


 




def cross_validation_test(df: pd.DataFrame, 
                     model: BaseModelWithCovariates,
                     test_model_call,     
                     cv,
                     model_name:str,
                     timeseries_kwargs: dict,
                     model_configuration: dict,            
                     verbose = False,
                     fold_start = 0, 
                     gpu_bool = gpus) -> tuple:
    """
    Function that cross validates a given model configuration and returns the mean of a set of performance
    metrics
    """
    start_cv = time.time()

    # Neptune logger initialisations
    if model_configuration.get("neptune_logger") is False:
            neptune_logging = False
    else: neptune_logging = True
    if neptune_logging:
        neptune_logger = neptune.init(
            project="MasterThesis/performance",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            name = model_name,
            description = "Cross validated using" + cv.get_type() + " with " + str(cv.n_splits) + " splits",
            source_files=["model_definitions.py", "run_lstm.py", "run_deepar.py", "performance_evaluation.py", "run_tft.py", "run_gru.py"]
        )  
        neptune_logger['Type']                                  = "Cross validated run"
       
        neptune_logger['Target']                                = timeseries_kwargs.get('target')
        neptune_logger['Architecture']                          = model_name
        neptune_logger['model/configuration']                   = model_configuration
        
        try:
            neptune_logger['model/hparams'] = model.hparams
        except:
            print(">No hparams")

    else: neptune_logger = None



    
    # Initial values and dictionary structures
    total_performance = {
        'mean_test_mae'   : [],
        'mean_test_smape' : [],
        'mean_test_rmse'  : [],
        'mean_test_loss'  : [],
        'mean_train_loss' : [],
        'mean_train_mae'  : [],
        'mean_train_rmse' : [],
        'mean_train_smape' : []
    }
    all_fold_data    = {}


    print(colored("PRED LENGTH{}".format(prediction_length), "green"))


    # Select fold to start on 
    if fold_start != 0:
        selected_fold = [fold for fold in cv.split(series_df_shifted)][fold_start]
        i = fold_start
        def folds_yield():
                    yield(selected_fold[0], selected_fold[1], selected_fold[2])
        folds = folds_yield()

    else: 
        folds = cv.split(df)
        i = 0


    # Initial increment value
    
    print(colored(">" + cv.get_type() + " cross validation pending... ", "green"),  colored("[", "green"), colored(i, "green"), colored("out of", "green"), colored(cv.get_n_splits(), "green"), colored("]", "green"))
    # Iterate through cross validation nested-list
    kwargs = {'model_name': model_name}
    for train_indices, val_indices, test_indices in folds:

        # Start timer per iteration session
        start_iter = time.time()
        print(colored(">>>>> Fold number:" + str(i), 'green'))
        
        # Record predictions per fold
        actuals_and_fc = pd.DataFrame()

        # Record performance per fold
        fold_performance = {
        'mean_test_mae'  : [],
        'mean_test_smape' : [],
        'mean_test_rmse' : [],
        'mean_train_loss' : [],
        'mean_train_mae' : [],
        'mean_train_rmse' : [],
        'mean_train_smape' : []
        }

        complete_folds =  math.floor(len(test_indices)/prediction_length) # Complete validation folds
        print(colored("> Starting walk-forward forecasts on test set", "cyan"))
        if verbose: print(colored("> Total number of folds  {}".format(complete_folds), "cyan"))
        for j in range(1, complete_folds+1):
            
            new_train_indices, new_val_indices, new_test_indices = overlapping_train_val_folds(j, prediction_length, train_indices, val_indices, test_indices)
            
            new_actuals_and_fc, train_metrics = test_model_call(series_df_shifted, model, model_configuration, timeseries_kwargs, new_train_indices, new_val_indices, new_test_indices, **kwargs )
            
             # Save evaluation forecasts
            if not os.path.exists("evaluation_forecasts"):
                os.makedirs("evaluation_forecasts")
            if not os.path.exists("evaluation_forecasts/"+model_name):
                os.makedirs("evaluation_forecasts/"+ model_name)

            #new_actuals_and_fc['datetime'] = remove_nans(series_df, 'linear').iloc[new_test_indices]['datetime'].tolist()
            actuals_and_fc = pd.concat([actuals_and_fc, new_actuals_and_fc])

            print("> Preds and FC: {}".format(new_actuals_and_fc))


            #Calculate metrics
            val_metrics = calc_metrics(new_actuals_and_fc)

            # Plot predictions
            train_make_plot(new_actuals_and_fc.actuals, new_actuals_and_fc.forecasts, save ="_{}_fold_{}_val_{}".format(model_name, i+1,j), legend_text= "MAE: {:.3f}".format(val_metrics['test_MAE']), neptune_logger=neptune_logger)
                

            # Interim step for total performance calculation 
            total_performance['mean_test_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            total_performance['mean_test_smape'].append(val_metrics['test_SMAPE'])
            total_performance['mean_test_rmse'].append(val_metrics['test_RMSE'])
            total_performance['mean_train_loss'].append(train_metrics.get('train_loss_epoch'))
            total_performance['mean_train_rmse'].append(train_metrics.get('train_RMSE_epoch'))
            total_performance['mean_train_smape'].append(train_metrics.get('train_SMAPE_epoch'))
            total_performance['mean_train_mae'].append(train_metrics.get('train_MAE_epoch'))


            fold_performance['mean_test_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            fold_performance['mean_test_smape'].append(val_metrics['test_SMAPE'])
            fold_performance['mean_test_rmse'].append(val_metrics['test_RMSE'])
            fold_performance['mean_train_loss'].append(train_metrics.get('train_loss_epoch'))
            fold_performance['mean_train_rmse'].append(train_metrics.get('train_RMSE_epoch'))
            fold_performance['mean_train_smape'].append(train_metrics.get('train_SMAPE_epoch'))
            fold_performance['mean_train_mae'].append(train_metrics.get('train_MAE_epoch'))



            

            print(colored(">>> End of walkforward prediction {} of {} test metrics {} ".format(j, complete_folds, val_metrics), "green"))
            print(colored(">>> End of walkforward averge test mae so far {} ".format(mean(total_performance['mean_test_mae'])), "green"))
            #### END OF WALKFORWARD PREDICTION ###


        
        
        i += 1  # Index increment 
        actuals_and_fc.to_csv("evaluation_forecasts/{}/{}_fold{}.csv".format(model_name,model_name, i))
        neptune_logger['forecasts/fold{}'.format(i)].upload("evaluation_forecasts/{}/{}_fold{}.csv".format(model_name,model_name, i))

        fold_data = {
                        "train_indices": train_indices, 
                        "val_indices": val_indices, 
                        "test_indices": test_indices, 
                        'test_mae'    : mean(fold_performance['mean_test_mae']), 
                        'test_rmse'   : mean(fold_performance['mean_test_rmse']), 
                        'test_smape'  : mean(fold_performance['mean_test_smape']),
                        'train_loss'  : mean(fold_performance['mean_train_loss']),
                        'train_rmse'  : mean(fold_performance['mean_train_rmse']),
                        'train_mae'  :  mean(fold_performance['mean_train_mae']),
                        'train_smape'  : mean(fold_performance['mean_train_smape']),
                        }
    
        neptune_logger['model/fold_data/'+str(i)] = fold_data

        all_fold_data[i] = fold_data 

        end_iter = time.time() # End timer for iteration

        iter_time = end_iter - start_iter
        print(colored(">>> Iteration time used:", "red"), colored(str(round(iter_time,1)), "red"), colored("seconds", "red"))

        # Deleting and clearing gpu memory
        if gpu_bool == 1:    
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        # Deleting and clearing gpu memory
        gc.collect()
    # Calculate standard deviation
    total_performance['sd_test_mae']   =  np.std(total_performance['mean_test_mae'])
    total_performance['sd_test_smape'] =  np.std(total_performance['mean_test_smape'])
    total_performance['sd_test_rmse']  =  np.std(total_performance['mean_test_rmse'])
    
    
    #  Performance result :Calulate mean of all fold performance
    total_performance['mean_test_mae']    = mean(total_performance['mean_test_mae'] )
    total_performance['mean_test_smape']  = mean(total_performance['mean_test_smape'] )
    total_performance['mean_test_rmse']   = mean(total_performance['mean_test_rmse'] )



    # Calculate standard deviation
    total_performance['sd_train_mae']   =  np.std(total_performance['mean_train_mae'])
    total_performance['sd_train_loss']   =  np.std(total_performance['mean_train_loss'])
    total_performance['sd_train_smape'] =  np.std(total_performance['mean_train_smape'])
    total_performance['sd_train_rmse']  =  np.std(total_performance['mean_train_rmse'])
    
    
    #  Performance result :Calulate mean of all fold performance
    total_performance['mean_train_mae']    = mean(total_performance['mean_train_mae'] )
    total_performance['mean_train_loss']    = mean(total_performance['mean_train_loss'] )
    total_performance['mean_train_smape']  = mean(total_performance['mean_train_smape'] )
    total_performance['mean_train_rmse']   = mean(total_performance['mean_train_rmse'] )

    end_cv = time.time() # End time

    cv_time = end_cv - start_cv  # CV time used
    
    
    # Console message and results
    print(colored("\n"+ ">"+ cv.get_type()+ "cross Validation complete...", "blue"))
    print(colored(">>> Total time used:", "red"), colored(str(round(cv_time,1)), "red"), colored("seconds", "red"))
    print(colored(">>> Mean Performance Result:", "green"), colored(total_performance, "green"), colored("<<<", "green"))

    if neptune_logging:
        neptune_logger['cv/time'] = cv_time
        neptune_logger['model/total_performance']     = total_performance
        #neptune_logger['model/training_and_val_data'] = training_and_val_data
        neptune_logger.stop()



    manual_test_logs({'total_performance':total_performance, 'training and validation data': all_fold_data}, model_name)
    return total_performance, all_fold_data





"""


cv = get_cv_type('sliding_evaluation', 5, **{'val_size':114})
t = [t for t in cv.split(series_df)]
test_folds = [t[i][2] for i in range(0,5)]
test_indices = sum(test_folds, [])
df = series_df.iloc[test_indices]

len(t[2][2])



"""


"""
cv = get_cv_type('sliding_evaluation', 5, **{'val_size':114})
target = 'intraday_price'

run_benchmark(series_df, eq_dayahead_benchmark, gru_model_config, timeseries_kwargs, t[0][0], t[0][1], t[0][2][0:38] )

cross_validation_test(series_df, eq_dayahead_benchmark, run_benchmark, cv,  "TEST eq benchmark", timeseries_kwargs, gru_model_config,  verbose = True)




prediction_length  = 38

timeseries_kwargs = {
        'time_idx':                            'seq',
        'target':                              target, 
        'min_encoder_length' :                  1,  
        'max_encoder_length' :                 300,  
        'group_ids':                           ['time_groups'],
        'min_prediction_length':               max_prediction_length,
        'max_prediction_length':               max_prediction_length,
        'time_varying_known_reals' :           [],
        'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
        'time_varying_known_categoricals':     [],
        'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
        'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                'hour':         NaNLabelEncoder(add_nan = True), 
                                                'day':         NaNLabelEncoder(add_nan = True), 
                                                'month':        NaNLabelEncoder(add_nan = True), 
                                                'year':        NaNLabelEncoder(add_nan = True), 
                                                'covid':        NaNLabelEncoder(add_nan = True), 
                                                'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                'northsealink': NaNLabelEncoder(add_nan = True)
                                                },
        'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
        'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
        'batch_size':                           128,
        'num_workers':                           11,
        'scale':                                False} 
gru_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [11],  
        'max_epochs':                           [100], 
        'patience':                             [15],
        'cell_type':                            ['GRU'],
        'hidden_size':                          [128],
        'rnn_layers':                           [2],
        'loss':                                 [metrics.MAE()],   
        'dropout':                              [0.1],
        'min_delta':                            [0.05],
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
        'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
        #'categorical_groups':                   {},
        'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
        'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'output_size':                           [1],
        'print_predictions':                    [True],
        'target':                                [target],
        'gradient_clip_val'                     : [0.6],
        #'target_lags':                          {},
        #'logging_metrics':                      None
        'optimizer':                            ['ranger'],
        'learning_rate' :                       [0.001],
        'accumulate_grad_batches':              [4], 
        'use_best_checkpoint':                  [False], 
        'k_best_checkpoints' :                   [3],
        'reduce_on_plateau_patience':           [2],
        'reduce_on_plateau_reduction' :         [2.0],
        'model_call' :                         [RecurrentNetwork],
        'neptune_logger':                       [True], # Check this
        'swa_epoch_start':                        [12]
        }


t = [t for t in cv.split(series_df)]


gru_model_config = take_first_model(gru_model_config)

_, gru_model, filtered_params = setup_lstm(series_df, gru_model_config, timeseries_kwargs)


data_module = DataModuleTimeSeries(df = series_df, train_indices = t[0][0]+t[0][1], val_indices=t[0][1], test_indices= t[0][1]+t[0][2][0:38], **timeseries_kwargs)
        
trainer, model = train_model(data_module=data_module, model= gru_model, neptune_logger = None, gpu_bool=gpus,model_configuration=gru_model_config)
        
perform_predictions(model, data_module.test_dataloader())

series_df.iloc[t[0][2]][0:38][target]
data_module.test_dataloader().dataset.data.get('target')[0][-38:]
preds = model.predict(data_module.test_dataloader(), return_index = 0, mode = "prediction")

len(data_module.test_dataloader().dataset.data.get('target')[0][0:38])




"""




#### Non-fitted benchmarks

def run_eq_intraday():

        prediction_length  = 38
        target = 'intraday_price'
        cross_validation_test(series_df_shifted, eq_short_term_dayahead_benchmark, evaluation_benchmarks, cv,  "EQ short-term intraday evaluation", {'target':target}, {'neptune_logger':True},  verbose = True)


def run_eq_dayahead():

        prediction_length  = 38
        target = 'dayahead_price'
        cross_validation_test(series_df_shifted, eq_short_term_dayahead_benchmark, evaluation_benchmarks, cv,  "EQ short-term dayahead evaluation", {'target':target}, {},  verbose = True)

def run_mean_dayahead():

        prediction_length  = 38
        target = 'dayahead_price'
        cross_validation_test(series_df_shifted, mean_benchmark, evaluation_benchmarks, cv,  "Mean dayahead evaluation", {'target':target}, {},  verbose = True)

def run_mean_intraday():

        prediction_length  = 38
        target = 'intraday_price'
        cross_validation_test(series_df_shifted, mean_benchmark, evaluation_benchmarks, cv,  "Mean intraday evaluation", {'target':target}, {},  verbose = True)


def run_take_last_dayahead():

        prediction_length  = 38
        target = 'dayahead_price'
        cross_validation_test(series_df_shifted, take_last_benchmark, evaluation_benchmarks, cv,  "Take last dayahead evaluation", {'target':target}, {},  verbose = True)


def run_take_last_intraday():

        prediction_length  = 38
        target = 'intraday_price'
        cross_validation_test(series_df_shifted, take_last_benchmark, evaluation_benchmarks, cv,  "Take last intraday evaluation", {'target':target}, {},  verbose = True)








def run_gru_intraday(fold_start = 0, name =  "GRU intraday evaluation"):




        prediction_length  = 38

        target = 'intraday_price'
        timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                  1,  
                'max_encoder_length' :                 300,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           128,
                'num_workers':                           7,
                'scale':                                False} 
        gru_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [7],  
                'max_epochs':                           [70], 
                'patience':                             [17],
                'cell_type':                            ['GRU'],
                'hidden_size':                          [256],
                'rnn_layers':                           [2],
                'loss':                                 [metrics.MAE()],   
                'dropout':                              [0.1],
                'min_delta':                            [0.05],
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                           [1],
                'print_predictions':                    [True],
                'target':                                [target],
                'gradient_clip_val'                     : [0.6],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'learning_rate' :                       [0.001],
                'accumulate_grad_batches':              [4], 
                'use_best_checkpoint':                  [False], 
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [2],
                'reduce_on_plateau_reduction' :         [2.0],
                'model_call' :                         [RecurrentNetwork],
                'neptune_logger':                       [True], # Check this
                'swa_epoch_start':                        [12]
                }



        gru_model_config = take_first_model(gru_model_config)

        _, gru_model, filtered_params = setup_lstm(series_df_shifted, gru_model_config, timeseries_kwargs)
        cross_validation_test(series_df_shifted, gru_model, evaluation_pytorch_models, cv, name, timeseries_kwargs, filtered_params, fold_start = fold_start, verbose = True)

def run_gru_dayahead():

    #cv = get_cv_type('sliding', 5, **{'val_size':114})

    prediction_length  = 38

    target = "dayahead_price"

    timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 336,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           128,
                'num_workers':                           7,
                'scale':                                False} 


    gru_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [7],  
                'max_epochs':                           [70], 
                'patience':                             [17],
                'cell_type':                            ['GRU'],
                'hidden_size':                          [256],
                'rnn_layers':                           [2],
                'loss':                                 [metrics.MAE()],   
                'dropout':                              [0.1],
                'min_delta':                            [0.05],
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                           [1],
                'print_predictions':                    [True],
                'target':                                [target],
                'gradient_clip_val'                     : [0.6],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'learning_rate' :                       [0.001],
                'accumulate_grad_batches':              [4], 
                'use_best_checkpoint':                  [False], 
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [2],
                'reduce_on_plateau_reduction' :         [2.0],
                'model_call' :                         [RecurrentNetwork],
                'neptune_logger':                       [True], # Check this
                'swa_epoch_start':                        [12]
                }

                
    gru_model_config = take_first_model(gru_model_config)

    _, gru_model, filtered_params = setup_lstm(series_df_shifted, gru_model_config, timeseries_kwargs)
    cross_validation_test(series_df_shifted, gru_model, evaluation_pytorch_models, cv,  "GRU dayahead evaluation", timeseries_kwargs, filtered_params,  verbose = True)

def run_lstm_dayahead(fold_start = 0, name =  "LSTM dayahead evaluation"):


    #cv = get_cv_type('sliding', 5, **{'val_size':114})

    prediction_length  = 38

    target = "dayahead_price"

    timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 336,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           128,
                'num_workers':                           7,
                'scale':                                False} 


    lstm_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [7],  
                'max_epochs':                           [70], 
                'patience':                             [17],
                'cell_type':                            ['LSTM'],
                'hidden_size':                          [128],
                'rnn_layers':                           [2],
                'loss':                                 [metrics.MAE()],   
                'dropout':                               [0.1],
                'min_delta':                             [0.05],
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                           [1],
                'print_predictions':                    [True],
                'target':                                [target],
                'gradient_clip_val'                     : [0.5],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'learning_rate' :                       [0.0001],
                'accumulate_grad_batches':              [2], 
                'use_best_checkpoint':                  [False], 
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [2],
                'reduce_on_plateau_reduction' :         [3.0],
                'model_call' :                          [RecurrentNetwork],
                'neptune_logger':                       [True],
                'swa_epoch_start':                      [15]
        }

    lstm_model_config = take_first_model(lstm_model_config)
    _, lstm_model, filtered_params = setup_lstm(series_df_shifted, lstm_model_config, timeseries_kwargs)
    cross_validation_test(series_df_shifted, lstm_model, evaluation_pytorch_models, cv, name, timeseries_kwargs, filtered_params, fold_start=fold_start, verbose = True)


def run_lstm_intraday():


    #cv = get_cv_type('sliding', 5, **{'val_size':114})

    prediction_length  = 38

    target = "intraday_price"

    timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 120,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           128,
                'num_workers':                           7,
                'scale':                                False} 

    lstm_model_config =   {
        'cell_type': 'LSTM',
        'batch_size':                           [64], 
        'num_workers':                          [7],  
        'max_epochs':                           [70], 
        'patience':                             [17],
        'cell_type':                            ['LSTM'],
        'hidden_size':                          [128],
        'rnn_layers':                           [2],
        'loss':                                 [metrics.MAE()],   
        'dropout':                              [0.1],
        'min_delta':                             [0.05],
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
        'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
        #'categorical_groups':                   {},
        'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
        'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'output_size':                          [1],
        'print_predictions':                    [True],
        'target':                                [target],
        'gradient_clip_val'                     : [0.005],
        #'target_lags':                          {},
        #'logging_metrics':                      None
        'optimizer':                            ['adamw'],
        'learning_rate' :                       [5.62341325190349e-04],
        'accumulate_grad_batches':              [2], 
        'use_best_checkpoint':                  [True], 
        'reduce_on_plateau_patience':           [2],
        'reduce_on_plateau_reduction' :         [2.0],
        'model_call' :                         [RecurrentNetwork],
        'neptune_logger':                       [True],
        'swa_epoch_start':                        [12]
        }

    lstm_model_config = take_first_model(lstm_model_config)
    _, lstm_model, filtered_params = setup_lstm(series_df_shifted, lstm_model_config, timeseries_kwargs)
    cross_validation_test(series_df_shifted, lstm_model, evaluation_pytorch_models, cv,  "LSTM intraday evaluation", timeseries_kwargs, filtered_params,  verbose = True)



def run_deepar_intraday():
    #cv = get_cv_type('sliding', 5, **{'val_size':114})

    prediction_length  = 38

    target = "intraday_price"

    timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 120,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           64,
                'num_workers':                          6,
                'scale':                                False} 

    deepar_model_config =   {
                'batch_size':                           [64], 
                'num_workers':                          [6],  
                'max_epochs':                           [70], 
                'patience':                             [17],
                'cell_type':                            ['LSTM'],
                'hidden_size':                          [128],
                'rnn_layers':                           [2],
                'dropout':                              [0.1],
                'min_delta':                            [0.03],
                #Multihorizon metrics should not be used
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                          [1],
                'print_predictions':                    [True],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['adamw'],
                'gradient_clip_val'                     :[0.05],
                'learning_rate' :                       [0.0001],
                'accumulate_grad_batches':              [2], 
                'use_best_checkpoint':                  [True],
                'reduce_on_plateau_patience':           [3],
                'reduce_on_plateau_reduction' :         [3.0],
                'model_call' :                         [DeepAR],
                'neptune_logger':                       [True],
                'swa_epoch_start' :                    [13] 
        }
    deepar_model_config = take_first_model(deepar_model_config)
    _, deepar_model, filtered_params = setup_deepar(series_df_shifted, deepar_model_config, timeseries_kwargs)
    cross_validation_test(series_df_shifted, deepar_model, evaluation_pytorch_models, cv,  "Deep AR intraday evaluation", timeseries_kwargs, filtered_params,  verbose = True)
           

def run_deepar_dayahead():
    #cv = get_cv_type('sliding', 5, **{'val_size':114})

    prediction_length  = 38

    target = "dayahead_price"

    timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 200,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               prediction_length,
                'max_prediction_length':               prediction_length,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           64,
                'num_workers':                          6,
                'scale':                                False} 

    deepar_model_config =   {
                'batch_size':                           [64], 
                'num_workers':                          [6],  
                'max_epochs':                           [70], 
                'patience':                             [16],
                'cell_type':                            ['GRU'],
                'hidden_size':                          [128],
                'rnn_layers':                           [2],
                'dropout':                              [0.1],
                'min_delta':                            [0.05],
                #Multihorizon metrics should not be used
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                          [1],
                'print_predictions':                    [True],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'gradient_clip_val'                     :[0.6],
                'learning_rate' :                       [0.001],
                'accumulate_grad_batches':              [5], 
                'use_best_checkpoint':                  [False],
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [2],
                'reduce_on_plateau_reduction' :         [2.0],
                'model_call' :                         [DeepAR],
                'neptune_logger':                       [True],
                'swa_epoch_start' :                    [15] 
        }
        

    deepar_model_config = take_first_model(deepar_model_config)
    _, deepar_model, filtered_params = setup_deepar(series_df_shifted, deepar_model_config, timeseries_kwargs)
    cross_validation_test(series_df_shifted, deepar_model, evaluation_pytorch_models, cv,  "Deep AR dayahead evaluation", timeseries_kwargs, filtered_params,  verbose = True)






def run_ets_intraday():
    
    target = 'intraday_price'
    prediction_length  = 38
    timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                  1,  
            'max_encoder_length' :                 336,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               prediction_length,
            'max_prediction_length':               prediction_length,
            'time_varying_known_reals' :           [],
            'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
            'time_varying_known_categoricals':     [],
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
            'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                    'hour':         NaNLabelEncoder(add_nan = True), 
                                                    'day':         NaNLabelEncoder(add_nan = True), 
                                                    'month':        NaNLabelEncoder(add_nan = True), 
                                                    'year':        NaNLabelEncoder(add_nan = True), 
                                                    'covid':        NaNLabelEncoder(add_nan = True), 
                                                    'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                    'northsealink': NaNLabelEncoder(add_nan = True)
                                                    },
            'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
            'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
            'batch_size':                           128,
            'num_workers':                           7,
            'scale':                                False} 

   
    ets_config_opt = {
        'error': 'add',
        'trend': None,
        'damped_trend': False,
        'seasonal': None,
        'seasonal_periods': 12
    }


    cross_validation_test(series_df_shifted, ets_model,evaluation_benchmark_models, cv, 'ETS evaluation intraday', timeseries_kwargs, ets_config_opt, verbose = True )






def run_ets_dayahead():

    target = 'dayahead_price'
    prediction_length  = 38
    timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                  1,  
            'max_encoder_length' :                 300,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               prediction_length,
            'max_prediction_length':               prediction_length,
            'time_varying_known_reals' :           [],
            'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
            'time_varying_known_categoricals':     [],
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
            'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                    'hour':         NaNLabelEncoder(add_nan = True), 
                                                    'day':         NaNLabelEncoder(add_nan = True), 
                                                    'month':        NaNLabelEncoder(add_nan = True), 
                                                    'year':        NaNLabelEncoder(add_nan = True), 
                                                    'covid':        NaNLabelEncoder(add_nan = True), 
                                                    'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                    'northsealink': NaNLabelEncoder(add_nan = True)
                                                    },
            'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
            'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
            'batch_size':                           128,
            'num_workers':                           7,
            'scale':                                False} 

   
    ets_config_opt = {
        'error': 'add',
        'trend': 'add',
        'damped_trend': True,
        'seasonal': 'add',
        'seasonal_periods': 168
    }


    cross_validation_test(series_df_shifted, ets_model,evaluation_benchmark_models, cv, 'ETS evaluation dayahead', timeseries_kwargs, ets_config_opt, verbose = True )

def run_arima_dayahead():

    target = 'dayahead_price'
    prediction_length  = 38
    timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                  1,  
            'max_encoder_length' :                 300,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               prediction_length,
            'max_prediction_length':               prediction_length,
            'time_varying_known_reals' :           [],
            'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
            'time_varying_known_categoricals':     [],
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
            'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                    'hour':         NaNLabelEncoder(add_nan = True), 
                                                    'day':         NaNLabelEncoder(add_nan = True), 
                                                    'month':        NaNLabelEncoder(add_nan = True), 
                                                    'year':        NaNLabelEncoder(add_nan = True), 
                                                    'covid':        NaNLabelEncoder(add_nan = True), 
                                                    'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                    'northsealink': NaNLabelEncoder(add_nan = True)
                                                    },
            'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
            'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
            'batch_size':                           128,
            'num_workers':                           7,
            'scale':                                False} 

   
    
    arima_orders_dayahead= {

        'p': 5,
        'd': 1,
        'q': 5,
        'P': 0,
        'D': 0,
        'Q': 0,
        's':0
    }



    cross_validation_test(series_df_shifted, arima_model,evaluation_benchmark_models, cv, 'ARIMA evaluation dayahead', timeseries_kwargs, arima_orders_dayahead, verbose = True )



def run_arima_intraday():

    target = 'intraday_price'
    prediction_length  = 38
    timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                  1,  
            'max_encoder_length' :                 300,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               prediction_length,
            'max_prediction_length':               prediction_length,
            'time_varying_known_reals' :           [],
            'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
            'time_varying_known_categoricals':     [],
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
            'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                    'hour':         NaNLabelEncoder(add_nan = True), 
                                                    'day':         NaNLabelEncoder(add_nan = True), 
                                                    'month':        NaNLabelEncoder(add_nan = True), 
                                                    'year':        NaNLabelEncoder(add_nan = True), 
                                                    'covid':        NaNLabelEncoder(add_nan = True), 
                                                    'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                    'northsealink': NaNLabelEncoder(add_nan = True)
                                                    },
            'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
            'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
            'batch_size':                           128,
            'num_workers':                           7,
            'scale':                                False} 

   
    
    arima_orders_intraday= {

        'p': 2,
        'd': 1,
        'q': 4,
        'P': 0,
        'D': 0,
        'Q': 0,
        's':0
    }



    cross_validation_test(series_df_shifted, arima_model,evaluation_benchmark_models, cv, 'ARIMA evaluation intraday', timeseries_kwargs, arima_orders_intraday, verbose = True )





if __name__ == '__main__':


        # Initialise cv

        #run_arima_dayahead()
        #run_arima_intraday()
        #run_ets_dayahead()
        run_ets_intraday()
        #run_deepar_intraday()
        #run_deepar_dayahead()


        #run_lstm_intraday()
        #run_lstm_dayahead(4,  "LSTM dayahead evaluation folds 5")
        #run_gru_intraday(4,  "GRU  intraday evaluation folds 5")
        #run_gru_dayahead()


        # Benchmark models
        #run_eq_dayahead()
        #run_eq_intraday()
        #run_mean_dayahead()
        #run_mean_intraday()
        #run_take_last_dayahead()
        #run_take_last_intraday()

        #run_ets_intraday()
        #run_ets_dayahead()
        #run_arima_dayahead()
        #run_arima_intraday()