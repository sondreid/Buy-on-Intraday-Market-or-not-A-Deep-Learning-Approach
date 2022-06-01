from cgi import test
from pytorch_forecasting import QuantileLoss, RecurrentNetwork, TemporalFusionTransformer
from model_definitions import *
from utilities import shift_covariates
from TFTtuning import *
from RNNtuning import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd



from model_definitions import cross_validation
import tensorflow as tf


# Set Cross validation type
cv = get_cv_type('sliding', 5)


target                                  = 'intraday_price'

robust_scaler = RobustScaler()
timeseries_kwargs = {
    'time_idx':                            'seq',
    'target':                              target, 
    'max_encoder_length' :                 200,  
    'group_ids':                           ['time_groups'],
    'max_prediction_length':               30,
    'time_varying_known_reals' :           time_varying_known_reals,
    'time_varying_unknown_reals':          time_varying_unknown_reals,
    'time_varying_known_categoricals':     time_varying_known_categoricals,
    'time_varying_unknown_categoricals':   time_varying_unknown_categoricals, 
    'static_categoricals':                 ['time_groups'],
    
    'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                            'hour':         NaNLabelEncoder(add_nan = True), 
                                            'month':        NaNLabelEncoder(add_nan = True), 
                                            'year':        NaNLabelEncoder(add_nan = True), 
                                            'covid':        NaNLabelEncoder(add_nan = True), 
                                            'nordlink':     NaNLabelEncoder(add_nan = True), 
                                            'northsealink': NaNLabelEncoder(add_nan = True)
                                            },
    'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
    'add_encoder_length':                  True,
    'add_relative_time_idx':               True,
    'batch_size':                           128,
    'num_workers':                           10,
    'scale':                                False
    }

def setup_tft(df, tft_model_config, timeseries_kwargs):
    """
    Setting up common parameters for TFT
    """
    

    # Parameters passed to trainer and other parameters
    tft_model_parameters =   [
                    'accumulate_grad_batches'
                    'max_epochs',
                    'gradient_clip_val',
                    'patience',
                    'auto_scale_batch_size']

    # Parameters passed to TemporalFusionTransformers                    
    tft_configuration =  [
        'hidden_size',
        'lstm_layers'
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
        'reduce_on_plateau_patience',
        'reduce_on_plateau_reduction',
        'reduce_on_plateau_min_lr',
        'optimizer_params',
        'optimizer']
    #timeseries_extra_params = ['num_workers', 'batch_size']
    timeseries_kwargs['num_workers'] = tft_model_config.get('num_workers')
    timeseries_kwargs['batch_size'] = tft_model_config.get('batch_size')
    filtered_tft_model_parameters = {key: dict_item for key, dict_item in tft_model_config.items() if key.startswith(tuple(tft_model_parameters))}
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





def run_tft_model(df : pd.DataFrame, tft_model_config:dict, timeseries_kwargs:dict):
    """
    Wrapper function for running cross validated TFT run
    """
    _, tft_model, filtered_tft_model_parameters = setup_tft(df, tft_model_config, timeseries_kwargs)
    cross_validation(df = df, model = tft_model, cv = cv, model_name = "TFT",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_tft_model_parameters)






if __name__ == "__main__":

    def run_grid_search():

        tft_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [10],  
        'max_epochs':                           [70], 
        'patience':                             [12],
        'hidden_size':                          [40, 50, 75],
        'lstm_layers':                          [1,3,5], 
        'dropout':                              [0.3, 0.6],
        'output_size':                          [7],
        'loss':                                 [QuantileLoss()],                
        'attention_head_size':                  [7],
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
        'hidden_continuous_size':               [19],
        #'hidden_continuous_sizes':              {},
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},    
        'learning_rate':                        [0.15],
        'gradient_clip_val':                    [0.0008198106964873125],
        'log_interval':                         [3],
        'log_val_interval':                     [3],
        #'log_gradient_flow':                    False,
        'reduce_on_plateau_patience':           [6],
        'reduce_on_plateau_reduction' :         [5],
        #'monotone_constaints':                  {},
        #'share_single_variable_networks':       False,
        #'logging_metrics':                      None
        #'optimizer_params': [{'lr':0.1}],
         'optimizer':  ['ranger', 'adamw']
        }


        grid_search_extended(series_df, run_tft_model, timeseries_kwargs, tft_model_config)


    def run_refit():
        # Refitting 
        timeseries_kwargs = {
            'time_idx':                            'seq',
            'target':                              target, 
            'max_encoder_length' :                 100,  
            'group_ids':                           ['time_groups'],
            'max_prediction_length':               24,
            'time_varying_known_reals' :           time_varying_known_reals,
            'time_varying_unknown_reals':          time_varying_unknown_reals + [target],
            'time_varying_known_categoricals':     time_varying_known_categoricals,
            'time_varying_unknown_categoricals':   time_varying_unknown_categoricals, 
            'static_categoricals':                 ['time_groups'],
            'add_relative_time_idx':               True,
            'categorical_encoders':               {'week':         NaNLabelEncoder(add_nan = True), 
                                                    'month':        NaNLabelEncoder(add_nan = True), 
                                                    'covid':        NaNLabelEncoder(add_nan = True), 
                                                    'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                    'northsealink': NaNLabelEncoder(add_nan = True)
                                                    },
            'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
            'add_encoder_length':                  True
            }

        tft_model_config =   {
            'batch_size':                           128, 
            'num_workers':                          9,  
            'max_epochs':                           70, 
            'patience':                             6,
            'hidden_size':                          15,
            'lstm_layers':                          5, 
            'dropout':                              0.28472627877645074,
            'output_size':                         7,
            'loss':                                 QuantileLoss(),                
            'attention_head_size':                  8,
            'min_delta': 1,
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
            'hidden_continuous_size':               15,
            #'hidden_continuous_sizes':              {},
            #'embedding_sizes':                      {},
            #'embedding_paddings':                   [],
            #'embedding_labels':                     {},    
            'learning_rate':                        0.05,
            'gradient_clip_val':                    0.09294355083997849,
            'log_interval':                         3,
            'log_val_interval':                     3,
            #'log_gradient_flow':                    False,
            'reduce_on_plateau_patience':           5,
            #'monotone_constaints':                  {},
            'reduce_on_plateau_min_lr':             0.00001, 
            #'share_single_variable_networks':       False,
            #'logging_metrics':                      None,
            'optimizer':  'adamw'
            }
        trainer, model = refit_tft(series_df, 'tft_refit_2', tft_model_config, timeseries_kwargs)

    ############# Run either grid search or refit
    #run_refit()
    run_grid_search()
