
from pytorch_forecasting import QuantileLoss, RecurrentNetwork, TemporalFusionTransformer
from model_definitions import *
from utilities import shift_covariates
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd



from model_definitions import cross_validation



time_varying_known_categoricals         = ['day', 'hour', 'year', 'week', 'month', 'nordlink', 'northsealink']
time_varying_unknown_categoricals       = ['covid']

# Combining danish power sources

series_df_shifted['DK1_combined_production_forecast'] =  series_df_shifted['DK1 Wind Power Production MWh/h 15min Forecast : ecsr-ens']+ series_df_shifted[ 'DK1 Solar Photovoltaic Production MWh/h 15min Forecast : ecsr-ens']



target = 'dayahead_price'

time_varying_known_reals_tft_1    =  ['DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'DK1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix' ,
                                'NO1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix', 
                                'NO5>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'NL>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'GB>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'SE3>NO1 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                
                                'DK1_combined_production_forecast',


                                'NO2 Consumption Index Chilling % 15min Forecast : ecsr-ens',
                                'NO2 Consumption Index Cloudiness % 15min Forecast : ecsr-ens',
                                'NO2 Consumption Index Heating % 15min Forecast : ecsr-ens',
                                'NO2 Consumption MWh/h 15min Forecast : ecsr-ens',
                                'NO2 Consumption Temperature °C 15min Forecast : ecsr-ens',
                                'NO2 Consumption MWh/h 15min Forecast : ecsr-ens',
                

                                 'NO1 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'NO5 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'DK1 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'NO2 Residual Load MWh/h 15min Forecast : ecsr-ens',

                                 'NO2 Hydro Reservoir Capacity Available MW REMIT : non-forecast',
                                 'NO2 Hydro Run-of-river Capacity Available MW REMIT : non-forecast'
                                ] + capacity_vars
time_varying_unknown_reals_tft_1       =  ['dk_power_non_wind',
                                            'NO2 Hydro Precipitation Energy MWh H Synthetic : non-forecast',
                                            'NO2 Hydro Reservoir Production MWh/h H Synthetic : non-forecast',
                                            'NO2 CHP Power Production MWh/h H Actual : non-forecast',
        
                                            'NO2 Price Regulation Down EUR/MWh H Actual : non-forecast',
                                            'NO2 Price Regulation Up EUR/MWh H Actual : non-forecast',
                                            'dayahead_price',
                                            'dayahead_volume',
                                            'intraday_price',
                                            'intraday_volume',
] + lag_vars

time_varying_known_reals_tft_2    =  ['DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'DK1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix' ,
                                'NO1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix', 
                                'NO5>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'NL>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                                'GB>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                        
                                'DK1_combined_production_forecast',
                
                                 'NO1 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'NO5 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'DK1 Residual Load MWh/h 15min Forecast : ecsr-ens',
                                 'NO2 Residual Load MWh/h 15min Forecast : ecsr-ens',

                                ] 
time_varying_unknown_reals_tft_2       =  ['dk_power_non_wind',
                                            'NO2 Hydro Precipitation Energy MWh H Synthetic : non-forecast',
                                            'NO2 Hydro Reservoir Production MWh/h H Synthetic : non-forecast',
                                            'dayahead_price',
                                            'dayahead_volume',
                                            'intraday_price',
                                            'intraday_volume',
]











#series_df = series_df[ ['datetime','seq','time_groups']  + time_varying_known_categoricals + time_varying_unknown_categoricals + time_varying_known_reals + time_varying_unknown_reals]
#series_df.loc[ ['datetime','seq','time_groups']  + time_varying_known_categoricals + time_varying_unknown_categoricals + time_varying_known_reals + time_varying_unknown_reals]



def setup_tft(df, tft_model_config, timeseries_kwargs):
    """
    Setting up common parameters for TFT
    """
    

    # Parameters passed to trainer and other parameters
    tft_model_parameters =   ['accumulate_grad_batches',
                    'max_epochs',
                    'gradient_clip_val',
                    'patience',
                    'min_delta',
                    'auto_scale_batch_size',
                    'swa_epoch_start',
                    'use_best_checkpoint',
                    'k_best_checkpoints',
                    'model_call', 
                    'neptune_logger',
                    'print_predictions']

    # Parameters passed to TemporalFusionTransformers                    
    tft_configuration =  [
        'hidden_size',
        'lstm_layers', 
        'dropout',
        'output_size',
        'loss',
        'attention_head_size',
        'max_encoder_length',
        'static_categoricals',
        'static_reals',
        'time_varying_categoricals_encoder',
        'time_varying_categoricals_decoder',
        'categorical_groups',
        'time_varying_reals_encoder'
        'time_varying_reals_decoder',
        'x_reals',
        'x_categoricals',
        'hidden_continuous_size',
        'embedding_sizes',
        'print_predictions',
        'embedding_paddings',
        'embedding_labels',
        'learning_rate',
        'log_interval',
        'log_val_interval',
        'log_gradient_flow',
        'reduce_on_plateau_patience',
        'monotone_constaints',
        'share_single_variable_networks',
        'logging_metrics',
        'reduce_on_plateau_reduction',
        'reduce_on_plateau_min_lr',
        'optimizer_params',
        'optimizer']


    #timeseries_extra_params = ['num_workers', 'batch_size']
    timeseries_kwargs['num_workers'] = tft_model_config.get('num_workers')
    timeseries_kwargs['batch_size'] = tft_model_config.get('batch_size')
    filtered_tft_model_parameters = {key: dict_item for key, dict_item in tft_model_config.items() if key.startswith(tuple(tft_model_parameters))}
    print(colored("Filtered parameters {}".format(filtered_tft_model_parameters, 'green')))
    filtered_tft_configuration = {key: dict_item for key, dict_item in tft_model_config.items() if key.startswith(tuple(tft_configuration))}
    try:
        data_module = DataModuleTimeSeries(
            df                                      = df, 
            **timeseries_kwargs)
        tft_model = TemporalFusionTransformer.from_dataset(
            dataset = data_module.train_dataloader().dataset, 
            **filtered_tft_configuration)
    except:
        raise ValueError("Missing parameters")
    return data_module, tft_model, filtered_tft_model_parameters


def test_tft(df, tft_model_config,timeseries_kwargs):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, tft_model, filtered_tft_model_parameters = setup_tft(df, tft_model_config, timeseries_kwargs)

    trainer, model = train_model(
        data_module = data_module, 
        model       = tft_model,  
        gpu_bool    = gpus, 
        logger = None,
        model_configuration = filtered_tft_model_parameters)
     #interpretations = variable_importance(model, data_module.test_dataloader())
    return trainer, model
   



def refit_tft(df :pd.DataFrame, model_name:str, tft_model_config:dict, timeseries_kwargs:dict):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, tft_model, filtered_tft_model_parameters = setup_tft(df, tft_model_config, timeseries_kwargs)

    neptune_logger = neptune.init(
        project         = "MasterThesis/Price",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        name = model_name,
        description = model_name, 
        source_files=["model_definitions.py", "model_implementations.py"]
    )  

    neptune_logger['Type']                                              = "Refit run"
    neptune_logger['Target']                                            = timeseries_kwargs.get('target')
    neptune_logger['Architecture']                                      = 'TFT'
    neptune_logger['model/configuration/tft_model_config'] = tft_model_config
    neptune_logger['model/configuration/timeseries_kwargs'] = timeseries_kwargs
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss', mode = 'min')
    neptune_logger['model/configuration']                               = tft_model_config
    kwargs = {'model_name': 'refit_tft', 'model_checkpoint':checkpoint_callback}
    trainer, model = train_model(
        data_module = data_module, 
        model       = tft_model,  
        gpu_bool    = gpus, 
        logger = neptune_logger,
        model_configuration = filtered_tft_model_parameters,
        **kwargs)

    neptune_logger.stop()
    # Manual logging
    print(colored("Saving model", 'cyan'))
    save_configurations(tft_model_config, timeseries_kwargs, model_name)
    if not os.path.exists("refit_runs"):
        os.makedirs("refit_runs")
    torch.save(model.state_dict(), 'refit_runs/'+model_name+'_model_weights.pth')
    #torch.save(model, 'refit_runs/'+model_name+'_model.pth')
    trainer.save_checkpoint('refit_runs/'+ model_name+ '_trainer.cpkt')
    return trainer, model




def run_tft_model(df : pd.DataFrame, tft_model_config:dict, timeseries_kwargs:dict, verbose = False):
    """
    Wrapper function for running cross validated TFT run
    """
    _, tft_model, filtered_tft_model_parameters = setup_tft(df, tft_model_config, timeseries_kwargs)
    cross_validation(df = df, model = tft_model, cv = cv, model_name = "TFT",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_tft_model_parameters, verbose = verbose)


### Solely for testing


def overlap_test():
    train_indices = np.arange(0, 100).tolist()
    val_indices = np.arange(80, 120).tolist()
    test_indices = np.arange(101, 200).tolist()
    data_module = DataModuleTimeSeries(df = series_df_small, train_indices = train_indices, val_indices=val_indices, test_indices=test_indices, **timeseries_kwargs)

    _, tft_model, filtered_tft_model_parameters = setup_tft(series_df_small, tft_model_config, timeseries_kwargs)
    trainer = pl.Trainer()
    trainer.fit(tft_model, data_module)
    trainer.test(tft_model, data_module.test_dataloader())


def refit_test():
    
    data_module, tft_model, filtered_tft_model_parameters = setup_tft(series_df_small, tft_model_config, timeseries_kwargs)
    trainer, model = test_tft(series_df_small, tft_model_config, timeseries_kwargs)


    # Testing whether additional fit yields perfomance increaase


    # Test set 
    test_df = series_df_shifted.iloc[19112:19212]
    test_df.year = test_df.year.astype(str)

    data_module.set_test_dataloader(test_df)
    trainer.test(model = model, dataloaders = data_module.test_dataloader())

    # Add new data

    new_df = series_df_shifted.iloc[19112:19180]
    new_data_module, tft_model, filtered_tft_model_parameters = setup_tft(new_df, tft_model_config, timeseries_kwargs)
    new_data_module = DataModuleTimeSeries(
                df                                      = new_df, 
                **timeseries_kwargs)
    trainer.fit(tft_model, new_data_module)


    # new test
    trainer.test(model = model, dataloaders = data_module.test_dataloader())


    new_df = series_df_shifted.iloc[19180:19212]
    new_data_module, tft_model, filtered_tft_model_parameters = setup_tft(new_df, tft_model_config, timeseries_kwargs)
    new_data_module = DataModuleTimeSeries(
                df                                      = new_df, 
                **timeseries_kwargs)

    trainer.fit_loop.max_epochs += 5
    trainer.fit_loop.max_steps +=5
    trainer.fit_loop.epoch_loop.max_steps += 5

    trainer.fit(tft_model, new_data_module)

    # new test
    trainer.test(model = model, dataloaders = data_module.test_dataloader())



if __name__ == "__main__":

    #robust_scaled_vars = {var: RobustScaler() for var in time_varying_known_reals + time_varying_unknown_reals if var not in target}
    def run_grid_search():
        timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                 1,  
            'max_encoder_length' :                 96,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               max_prediction_length,
            'max_prediction_length':               max_prediction_length+2,
            'time_varying_known_reals' :           time_varying_known_reals_tft_2,
            'time_varying_unknown_reals':          time_varying_unknown_reals_tft_2,
            'time_varying_known_categoricals':     time_varying_known_categoricals,
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals, 
            'static_categoricals':                 ['time_groups'],

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
            'add_encoder_length':                  True,
            'add_relative_time_idx':               True,
            'scalers': {var: RobustScaler() for var in time_varying_known_reals + time_varying_unknown_reals if var not in target},
            'batch_size':                           64,
            'num_workers':                           7,
            'scale':                                False
            }

        tft_model_config =   {
        'batch_size':                           [64], 
        'num_workers':                          [7],  
        'max_epochs':                           [70], 
        'patience':                             [7],
        'hidden_size':                          [128],
        'lstm_layers':                          [3], 
        'dropout':                              [0.05],
        'output_size':                          [1],
        'loss':                                 [metrics.MAE()],                
        'attention_head_size':                  [4],
        'min_delta':                            [0.1],
        #'max_encoder_length':                   10,
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        #'time_varying_categoricals_encoder':    [],
        #'time_varying_categoricals_decoder':    [],
        #'categorical_groups':                   {},
        #'time_varying_reals_encoder':           [target],
        #'time_varying_reals_decoder':           [],
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'hidden_continuous_size':               [64],
        #'hidden_continuous_sizes':              {},
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},    
        'learning_rate':                        [0.05],
        'gradient_clip_val':                    [0.3],
        'log_interval':                         [5],
        'log_val_interval':                     [5],
        #'log_gradient_flow':                    False,
        'reduce_on_plateau_patience':           [2],
        'reduce_on_plateau_reduction' :         [2.0],
        #'monotone_constaints':                  {},
        #'share_single_variable_networks':       False,
        #'logging_metrics':                      None
        #'optimizer_params': [{'lr':0.1}],
        'optimizer':                           ['ranger'],
        'accumulate_grad_batches':               [3], 
        'use_best_checkpoint':                   [False],
        'k_best_checkpoints':                     [2],
        'neptune_logger':                        [True], # Check this
        'swa_epoch_start':                        [10],
        'model_call' : [TemporalFusionTransformer]
        }


        grid_search_extended(series_df_shifted, run_tft_model, timeseries_kwargs, tft_model_config, verbose=True)




    ############# Run either grid search or refit
    #run_refit()
    run_grid_search()
