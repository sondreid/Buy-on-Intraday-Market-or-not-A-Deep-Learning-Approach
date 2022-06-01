from run_lstm import * 


"""

Script for running GRU


"""



target = "dayahead_price"

lag_vars_expanded = capacity_vars = [covariate for covariate in series_df_shifted.columns if "lag" in covariate]
# Specifications for TimeSeriesDataset


forecast_vars = [covariate for covariate in series_df_shifted.columns if "Forecast" in covariate or 'REMIT' in covariate]

capacity_vars = [covariate for covariate in series_df_shifted.columns if "Day-Ahead Capacity" in covariate]
""""

time_varying_known_categoricals         = ['day', 'hour', 'year', 'week', 'month', 'nordlink', 'northsealink']
time_varying_unknown_categoricals       = ['covid']

non_forecast_vars = [covariate for covariate in series_df.columns if covariate not in forecast_vars + capacity_vars + time_varying_known_categoricals + time_varying_unknown_categoricals + ['seq']+['datetime']+['time_groups']]
non_forecast_vars =[covariate for covariate in non_forecast_vars if 'price_lag' not in covariate]"""
time_varying_known_reals                = forecast_vars + capacity_vars
time_varying_unknown_reals              =  non_forecast_vars + lag_vars_expanded




def run_gru_model(df : pd.DataFrame, gru_model_config:dict, timeseries_kwargs:dict, verbose = False):
    """
    Wrapper function for running cross validated TFT run
    """
    _, gru_model, filtered_gru_model_parameters = setup_lstm(df, gru_model_config, timeseries_kwargs)
    cross_validation(df = df, model = gru_model, cv = cv, model_name = "GRU",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_gru_model_parameters, verbose = verbose)




"""

timeseries_kwargs = {
        'time_idx':                            'seq',
        'target':                              target, 
        'min_encoder_length' :                 1,  
        'max_encoder_length' :                 50,  
        'group_ids':                           ['time_groups'],
        'min_prediction_length':               max_prediction_length,
        'max_prediction_length':               max_prediction_length+2,
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
        'batch_size':                           50,
        'num_workers':                           7,
        'scale':                                False} 
gru_model_config =   {
        'cell_type': 'GRU',
        'batch_size':                           50, 
        'num_workers':                          11,  
        'max_epochs':                           3, 
        'patience':                             15,
        'cell_type':                            'GRU',
        'hidden_size':                          32,
        'rnn_layers':                           4,
        'loss':                                 metrics.MAE(),   
        'dropout':                              0.35,
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        'time_varying_categoricals_encoder':    time_varying_unknown_categoricals + time_varying_known_categoricals,
        'time_varying_categoricals_decoder':    time_varying_unknown_categoricals + time_varying_known_categoricals,
        #'categorical_groups':                   {},
        'time_varying_reals_encoder':           time_varying_unknown_reals + time_varying_known_reals,
        'time_varying_reals_decoder':           [x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals,
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'output_size':                          1,
        'target':target,
        #'target_lags':                          {},
        #'loss':                                 metrics.QuantileLoss(), # QuantileLoss does not work with recurrent network
        #'logging_metrics':                      None
        'gradient_clip_val':                      0.005,
        'optimizer':                            'adamw',
        'learning_rate' :                       5.62341325190349e-03,
        'accumulate_grad_batches':              2, 
        'use_best_checkpoint':                  False,
        'k_best_checkpoints':                    True, 
        'reduce_on_plateau_patience':           3,
        'reduce_on_plateau_reduction' :         2.0,
        'model_call' :                         RecurrentNetwork,
         'neptune_logger':                       False
        }


data_module, gru_model, filtered_gru_model_parameters = setup_lstm(series_df, gru_model_config, timeseries_kwargs)


#find_initial_learning_rate(gru_model, data_module)

#trainer, model = test_lstm(series_df_shifted, series_df_shifted.iloc[-200:-1], gru_model_config, timeseries_kwargs )

trainer, model = train_model(data_module, gru_model, None, gru_model_config)


preds = perform_predictions(model, data_module.val_dataloader())


ensemble_preds = ensemble_checkpoints(trainer, RecurrentNetwork, data_module.val_dataloader())

best_checkpoint = trainer.checkpoint_callback.best_k_models.get(list(trainer.checkpoint_callback.best_k_models.keys())[-1])

best_checkpoint_model = RecurrentNetwork.load_from_checkpoint(list(trainer.checkpoint_callback.best_k_models.keys())[-1])

best_checkpoint_preds = perform_predictions(best_checkpoint_model, data_module.val_dataloader())


print(calc_metrics(preds))
print(calc_metrics(ensemble_preds))
print(calc_metrics(best_checkpoint_preds))


mean_absolute_error(preds.actuals, ensemble_preds)
mean_absolute_error(preds.actuals, preds.forecasts)

mean_absolute_error(best_checkpoint_preds.actuals, best_checkpoint_preds.forecasts )


plot_predictions_simple(model, eval)
eval = remove_nans(series_df_evaluation.iloc[0:46],'linear')
preds = model.predict(remove_nans(series_df_evaluation.iloc[0:60],'linear'), return_index = 0, mode = "prediction")
perform_predictions(model, remove_nans(series_df_evaluation.iloc[0:60],'linear'), 0)
"""

def main():
    #robust_scaled_vars = {var: RobustScaler() for var in time_varying_known_reals + time_varying_unknown_reals if var not in target}
    def run_grid_search():
        
        target = "dayahead_price"
        timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 300,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               max_prediction_length,
                'max_prediction_length':               max_prediction_length+2,
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
                'batch_size':                           256,
                'num_workers':                           7,
                'scale':                                False} 

        gru_model_config =   {
                        'batch_size':                           [128], 
                        'num_workers':                          [7],  
                        'max_epochs':                           [70], 
                        'patience':                             [17],
                        'cell_type':                            ['GRU'],
                        'hidden_size':                          [256],
                        'rnn_layers':                           [3],
                        'loss':                                 [metrics.MAE()],   
                        'dropout':                              [0.3],
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

        target = 'intraday_price'
        timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 300,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               max_prediction_length,
                'max_prediction_length':               max_prediction_length+2,
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
                'batch_size':                           256,
                'num_workers':                           7,
                'scale':                                False} 

        gru_model_config =   {
                        'batch_size':                           [128], 
                        'num_workers':                          [7],  
                        'max_epochs':                           [70], 
                        'patience':                             [17],
                        'cell_type':                            ['GRU'],
                        'hidden_size':                          [256],
                        'rnn_layers':                           [3],
                        'loss':                                 [metrics.MAE()],   
                        'dropout':                              [0.3],
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

        grid_search_extended(series_df_shifted, run_gru_model, timeseries_kwargs, gru_model_config, verbose=True)
        



    ############# Run either grid search or refit
    #run_refit()
    run_grid_search()





if __name__ == "__main__":

    main()


