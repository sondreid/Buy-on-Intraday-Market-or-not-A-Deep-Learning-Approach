from model_definitions import * 



from pytorch_forecasting import NHiTS
from model_definitions import *
from utilities import shift_covariates
from TFTtuning import *
from RNNtuning import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from model_definitions import cross_validation
import tensorflow as tf




target                                  = 'intraday_price'



# Shifting the forecast h periods back in time in order to handle forecasted values
series_df_shifted = series_df_shifted.copy()
series_df_shifted[time_varying_known_reals] = shift_covariates(
    df      = series_df_shifted, 
    columns = time_varying_known_reals, 
    h       = -max_prediction_length)
    

cv = get_cv_type('sliding', 5, **{'val_size':240})
max_prediction_length  = 50


def setup_nhits(df, nhits_model_config, timeseries_kwargs):
    """
    Setting up common parameters for TFT
    """
    

    # Parameters passed to trainer and other parameters
    nhits_model_parameters =   ['accumulate_grad_batches',
                    'max_epochs',
                    'gradient_clip_val',
                    'patience',
                    'min_delta',
                    'auto_scale_batch_size',
                    'use_best_checkpoint',
                    'model_call',
                    'print_predictions',
                    'neptune_logger']

    # Parameters passed to TemporalFusionTransformers                    
    nhits_configuration =  [
        'hidden_size',
        'static_hidden_size '
        'dropout',
        'shared_weights ',
        'initialization',
        'output_size',
        'loss',
        'n_blocks',
        'pooling_sizes',
        'pooling_mode',
        'downsample_frequencies',
        'interpolation_mode',
        'batch_normalization',
        'prediction_length',
        'time_varying_categoricals_encoder',
        'time_varying_categoricals_decoder',
        'categorical_groups',
        'time_varying_reals_encoder',
        'time_varying_reals_decoder',
        'x_reals',
        'x_categoricals',
        'embedding_sizes',
        'embedding_paddings',
        'embedding_labels',
        'learning_rate',
        'log_interval',
        'log_val_interval',
        'log_gradient_flow',
        'monotone_constaints',
        'share_single_variable_networks',
        'logging_metrics',
        'reduce_on_plateau_patience',
        'reduce_on_plateau_reduction',
        'reduce_on_plateau_min_lr',
        'optimizer_params',
        'optimizer']


    #timeseries_extra_params = ['num_workers', 'batch_size']
    timeseries_kwargs['num_workers'] = nhits_model_config.get('num_workers')
    timeseries_kwargs['batch_size'] = nhits_model_config.get('batch_size')
    filtered_nhits_model_parameters = {key: dict_item for key, dict_item in nhits_model_config.items() if key.startswith(tuple(nhits_model_parameters))}
    print(colored("Filtered parameters {}".format(filtered_nhits_model_parameters, 'green')))
    filtered_nhits_configuration = {key: dict_item for key, dict_item in nhits_model_config.items() if key.startswith(tuple(nhits_configuration))}
    try:
        data_module = DataModuleTimeSeries(df  = df, **timeseries_kwargs)
        nhits_model = NHiTS.from_dataset(
            dataset = data_module.train_dataloader().dataset, 
            **filtered_nhits_configuration)
    except:
        raise ValueError("Missing parameters")
    return data_module, nhits_model, filtered_nhits_model_parameters



def test_nhits(df, nhits_model_config,timeseries_kwargs):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, nhits_model, filtered_tft_model_parameters = setup_nhits(df, nhits_model_config, timeseries_kwargs)

    trainer, model = train_model(
        data_module = data_module, 
        model       = nhits_model,  
        gpu_bool    = gpus, 
        neptune_logger =  None,
        model_configuration = filtered_tft_model_parameters)
     #interpretations = variable_importance(model, data_module.test_dataloader())
    return trainer, model



test_nhits(series_df_shifted, nhits_model_config, timeseries_kwargs)

#perform_predictions(model)



def refit_nhits(df :pd.DataFrame, model_name:str, tft_model_config:dict, timeseries_kwargs:dict):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, tft_model, filtered_tft_model_parameters = setup_nhits(df, tft_model_config, timeseries_kwargs)

    neptune_logger = neptune.init(
        project         = "MasterThesis/Price",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        name = model_name,
        description = model_name, 
        source_files=["model_definitions.py", "model_implementations.py"]
    )  

    neptune_logger['Type']                                              = "Refit run"
    neptune_logger['Target']                                            = timeseries_kwargs.get('target')
    neptune_logger['Architecture']                                      = model_name
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




def run_nhits_model(df : pd.DataFrame, nhits_model_config:dict, timeseries_kwargs:dict, verbose = False):
    """
    Wrapper function for running cross validated TFT run
    """
    _, nhits_model, filtered_tft_model_parameters = setup_nhits(df, nhits_model_config, timeseries_kwargs)
    cross_validation(df = df, model = nhits_model, cv = cv, model_name = "nhits",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_tft_model_parameters, verbose = verbose)







if __name__ == "__main__":






    ### nhits ##
    # Notes
    # Don't use load from best_checkpoint if Ctrl+C training
    def run_grid_search():

        timeseries_kwargs = {'time_idx':                            'seq',
            'target':                              target, 
            'min_encoder_length' :                 20,  
            'max_encoder_length' :                 20,  
            'group_ids':                           ['time_groups'],
            'min_prediction_length':               max_prediction_length+2,
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
            'batch_size':                           128,
            'num_workers':                           9,
            'scale':                                False} 

        nhits_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [11],  
        'max_epochs':                           [70], 
        'patience':                             [15],
        'hidden_size':                          [512],
        'static_hidden_size':                    [512],
        'shared_weights':                        [True],
        'initialization':                       ['lecun_normal'],
        #'n_blocks ':                            [],
        #'n_layers':                              [],
        #'pooling_sizes':                        [],
        #'pooling_mode' :                         [],
        #'downsample_frequencies' :              [],
        'interpolation_mode':                    ['linear'],
        'loss':                                 [MSE()],   
        'dropout':                              [0.2],
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        #'categorical_groups':                   {},
        'time_varying_reals_encoder':           [time_varying_unknown_reals],
        'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target]],
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'output_size':                          [1],
        'print_predictions':                    [True],
        #'logging_metrics':                      None
        'optimizer':                            ['adamw'],
        'learning_rate' :                       [0.0001],
        'accumulate_grad_batches':              [5], 
        'use_best_checkpoint':                  [True], 
        'reduce_on_plateau_patience':           [6],
        'reduce_on_plateau_reduction' :         [4.0],
        'model_call' :                         [NHiTS],
        'neptune_logger':                       [True]
        }
        grid_search_extended(series_df_shifted, run_nhits_model, timeseries_kwargs, nhits_model_config, verbose=True)

    def run_refit_nhits():
            nhits_model_config =   {
                'cell_type': 'nhits',
                'batch_size':                           64, 
                'num_workers':                          11,  
                'max_epochs':                           70, 
                'patience':                             15,
                'cell_type':                            'nhits',
                'hidden_size':                          512,
                'static_hidden_size':                    512,
                'shared_weights':                        True,
                'initialization':                       'lecun_normal',
                #'n_blocks ':                            [],
                #'n_layers':                              [],
                #'pooling_sizes':                        [],
                #'pooling_mode' :                         [],
                #'downsample_frequencies' :              [],
                'interpolation_mode':                    'linear',
                'prediction_length'                   : max_prediction_length,
                'loss':                                 MSE(),   
                'dropout':                              0.2,
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           time_varying_unknown_reals ,
                'time_varying_reals_decoder':           [x for x in time_varying_unknown_reals if x != target],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'print_predictions':                    True,
                'target':                                target,
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                           'adamw',
                'learning_rate' :                       0.01,
                'accumulate_grad_batches':              5, 
                'use_best_checkpoint':                  True, 
                'reduce_on_plateau_patience':           6,
                'reduce_on_plateau_reduction' :         4.0,
                'model_call' :                         NHiTS,
                'neptune_logger':                       True
                }



    run_grid_search()
