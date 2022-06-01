from platform import architecture
from pytorch_forecasting import QuantileLoss, RecurrentNetwork, TemporalFusionTransformer
from model_definitions import *
from utilities import shift_covariates
from TFTtuning import *
from RNNtuning import *
import pickle



###################### Parameters ###########################

# Definining target
target  = 'intraday_price'
# Number of cores in use
num_workers = 6
max_epochs                      = 50
n_trials                        = 5
patience                        = 6
timeout                         = 3600 * 8.0  # 8 hours
gradient_clip_val_range         = (0.0001, 1)
hidden_size_range               = (15, 100)
hidden_continuous_size_range    = (15, 100)
attention_head_size_range       = (3, 16)
dropout_range                   = (0.1, 0.6)
learning_rate_range             = (0.0001, 0.1)
use_learning_rate_finder        = False
trainer_kwargs                  = {'gpus':gpus, 'accelerator': 'gpu' if gpus == 1 else 'cpu', 'log_every_n_steps': 30}
study_var                       = None
verbose                         = None # 0: get_verbosity(), 1, warnings, 2. info, 3. debugging
pruner                          = optuna.pruners.SuccessiveHalvingPruner()


def neptune_logger_wrapper(architecture, study, data_module, configuration, timeseries_kwargs):
    """
    Conducts logging for a given architecture
    """

    neptune_logger = neptune.init(
            project         = "MasterThesis/Price",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            source_files    = ["model_definitions.py", "model_implementations.py", "random_search_hyperparameters.py"]
        )  

    # Neptune logging
    neptune_logger['Target']                                            = target
    neptune_logger['Type']                                              = "Random Search Hyperparameter Optimization"
    neptune_logger['Architecture']                                      = architecture
    neptune_logger['model/optimal_params']                              = study.best_params
    neptune_logger['model/configuration/batch_size']                    = data_module.batch_size
    neptune_logger['model/configuration/max_epochs']                    = max_epochs
    neptune_logger['model/configuration/n_trials']                      = n_trials
    neptune_logger['model/configuration/patience']                      = patience
    neptune_logger['model/configuration/timeout']                       = timeout  # 8 hours
    neptune_logger['model/configuration/gradient_clip_val_range']       = gradient_clip_val_range
    neptune_logger['model/configuration/hidden_size_range']             = hidden_size_range
    neptune_logger['model/configuration/hidden_continous_size_range']   = hidden_continuous_size_range
    neptune_logger['model/configuration/attention_head_size_range']     = attention_head_size_range
    neptune_logger['model/configuration/dropout_range']                 = dropout_range
    neptune_logger['model/configuration/learning_rate_range']           = learning_rate_range    
    neptune_logger['model/configuration/use_learning_rate_finder']      = use_learning_rate_finder
    neptune_logger['model/configuration/trainer_kwargs']                = trainer_kwargs
    neptune_logger['model/configuration/pruner']                        = pruner
    neptune_logger['model/configuration/special_configuration']         = configuration
    neptune_logger['model/configuration/timeseries_kwargs']             = timeseries_kwargs
    neptune_logger['train/loss/val_loss']                               = study.best_value


    neptune_logger.stop()
    
    if not os.path.exists("random_search"):
            os.makedirs("random_search")
    if not os.path.exists("random_search/"+ architecture):
            os.makedirs("random_search/"+ architecture)
    # save study results - also we can resume tuning at a later point in time
    with open("random_search/" + architecture + "/" + architecture + "_random_search_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

def main_tft():
    #################################################################################################################################
    ##### TFT #######################################################################################################################
    #################################################################################################################################


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

    # Testing data module
    data_module = DataModuleTimeSeries(
        df                                      = series_df_shifted, 
        batch_size                              = 128, 
        num_workers                             = num_workers, # adjust on line 17
        train_indices                           = None, 
        val_indices                             = None, 
        test_indices                            = None, 
        **timeseries_kwargs)


    tft_configuration =   {
        #'hidden_size':                          32,
        'lstm_layers':                          3, 
        #'dropout':                              0.2, # The 20% value is widely accepted as the best compromise between preventing model overfitting and retaining model accuracy. A good starting point is 20% but the dropout value should be kept small (up to 50%). Source:https://medium.com/geekculture/10-hyperparameters-to-keep-an-eye-on-for-your-lstm-model-and-other-tips-f0ff5b63fcd4
        #'output_size':                          7,
        #'loss':                                 QuantileLoss(),                
        #'attention_head_size':                  4,
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
        #'hidden_continuous_size':               8,
        #'hidden_continuous_sizes':              {},
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},    
        #'learning_rate':                        0.01, # Overwritten when using use_learning_rate_finder = True in tft_optimize_hyperparameters
        #'log_interval':                         -1,
        #'log_val_interval':                     None,
        #'log_gradient_flow':                    False,
        #'reduce_on_plateau_patience':           1000,
        #'monotone_constaints':                  {},
        #'share_single_variable_networks':       False,
        #'logging_metrics':                      None, 
    }


    # OPTIMIZE TFT MODEL
    study = tft_optimize_hyperparameters(
        train_dataloaders               = data_module.train_dataloader(),
        val_dataloaders                 = data_module.val_dataloader(),
        model_path                      = 'optuna_test',
        max_epochs                      = max_epochs,
        n_trials                        = n_trials,
        patience                        = patience,
        timeout                         = timeout,  # 8 hours
        gradient_clip_val_range         = gradient_clip_val_range,
        hidden_size_range               = hidden_size_range,
        hidden_continuous_size_range    = hidden_continuous_size_range,
        attention_head_size_range       = attention_head_size_range,
        dropout_range                   = dropout_range,
        learning_rate_range             = learning_rate_range,
        use_learning_rate_finder        = use_learning_rate_finder,
        trainer_kwargs                  = trainer_kwargs,
        log_dir                         = "tb_logs",
        study                           = study_var,
        verbose                         = verbose, # 0: get_verbosity(), 1, warnings, 2. info, 3. debugging
        pruner                          = pruner,
        **tft_configuration)
    

    neptune_logger_wrapper(architecture = 'TFT', study= study, data_module=data_module,configuration=tft_configuration, timeseries_kwargs=timeseries_kwargs)

    return study
 



#################################################################################################################################
##### LSTM ######################################################################################################################
#################################################################################################################################
def main_lstm():
    """
    LSTM model wrapper
    """
    lstm_config = common_rnn(target, num_workers)
    timeseries_kwargs = lstm_config.get('timeseries_kwargs')
    series_df_shifted = lstm_config.get('series_df_shifted')
    # Testing data module
    data_module = DataModuleTimeSeries(
        df = series_df_shifted, 
        batch_size = 256, 
        num_workers = num_workers, # adjust on line 17
        **timeseries_kwargs)
    try:
        lstm_configuration =   {
            'cell_type':                            'LSTM',
            #'hidden_size':                          32,
            'rnn_layers':                            4,
            #'dropout':                              0.1,
            #'static_categoricals':                  [],
            #'static_reals':                         [],
            'time_varying_categoricals_encoder':    lstm_config.get('time_varying_unknown_categoricals'),
            'time_varying_categoricals_decoder':    lstm_config.get('time_varying_unknown_categoricals'),
            #'categorical_groups':                   {},
            'time_varying_reals_encoder':           lstm_config.get('time_varying_unknown_reals') + [target],
            'time_varying_reals_decoder':           lstm_config.get('time_varying_unknown_reals'),
            #'embedding_sizes':                      {},
            #'embedding_paddings':                   [],
            #'embedding_labels':                     {},
            #'x_reals':                              [],
            #'x_categoricals':                       [],
            'output_size':                          1,
            'target':                               target,
            #'target_lags':                          {},
            #'loss':                                 metrics.QuantileLoss(), # QuantileLoss does not work with recurrent network
            #'logging_metrics':                      None, 
            }
    except:
        raise ValueError("Missing parameters")
    
    
    # Model
    lstm_model = RecurrentNetwork.from_dataset(
        dataset = data_module.train_dataloader().dataset, 
        **lstm_configuration)
    lstm_model.hparams



    # OPTIMIZE LSTM MODEL
    study = rnn_optimize_hyperparameters(
        train_dataloaders               = data_module.train_dataloader(),
        val_dataloaders                 = data_module.val_dataloader(),
        model_path                      = 'optuna_test',
        max_epochs                      = max_epochs,
        n_trials                        = n_trials,
        patience                        = patience,
        timeout                         = timeout,  # 8 hours
        gradient_clip_val_range         = gradient_clip_val_range,
        hidden_size_range               = hidden_size_range,
        dropout_range                   = dropout_range,
        learning_rate_range             = learning_rate_range,
        use_learning_rate_finder        = use_learning_rate_finder,
        trainer_kwargs                  = trainer_kwargs,
        log_dir                         = "tb_logs",
        study                           = study_var,
        verbose                         = verbose, # 0: get_verbosity(), 1, warnings, 2. info, 3. debugging
        pruner                          = pruner,
        **lstm_configuration)

    
    neptune_logger_wrapper(architecture = 'LSTM', study= study, data_module=data_module,configuration=lstm_configuration, timeseries_kwargs=timeseries_kwargs)
    return study


def main_gru():
    """
    GRU model wrapper
    """
    gru_config = common_rnn(target, num_workers)
    timeseries_kwargs = gru_config.get('timeseries_kwargs')
    series_df_shifted = gru_config.get('series_df_shifted')
    # Testing data module
    data_module = DataModule(
        df = series_df_shifted, 
        batch_size = 256, 
        num_workers = num_workers, # adjust on line 17
        **timeseries_kwargs)
    try:
        gru_configuration =   {
            'cell_type':                            'GRU',
            #'hidden_size':                          32,
            'rnn_layers':                            4,
            #'dropout':                              0.1,
            #'static_categoricals':                  [],
            #'static_reals':                         [],
            'time_varying_categoricals_encoder':    gru_config.get('time_varying_unknown_categoricals'),
            'time_varying_categoricals_decoder':    gru_config.get('time_varying_unknown_categoricals'),
            #'categorical_groups':                   {},
            'time_varying_reals_encoder':           gru_config.get('time_varying_unknown_reals') + [target],
            'time_varying_reals_decoder':           gru_config.get('time_varying_unknown_reals'),
            #'embedding_sizes':                      {},
            #'embedding_paddings':                   [],
            #'embedding_labels':                     {},
            #'x_reals':                              [],
            #'x_categoricals':                       [],
            'output_size':                          1,
            'target':                               target,
            #'target_lags':                          {},
            #'loss':                                 metrics.QuantileLoss(), # QuantileLoss does not work with recurrent network
            #'logging_metrics':                      None, 
            }
    except:
        raise ValueError("Missing parameters")
    
    
    
    
    # Model
    gru_model = RecurrentNetwork.from_dataset(
        dataset = data_module.train_dataloader().dataset, 
        **gru_configuration)
    gru_model.hparams
 

    # OPTIMIZE LSTM MODEL
    study = rnn_optimize_hyperparameters(
        train_dataloaders               = data_module.train_dataloader(),
        val_dataloaders                 = data_module.val_dataloader(),
        model_path                      = 'optuna_test',
        max_epochs                      = max_epochs,
        n_trials                        = n_trials,
        patience                        = patience,
        timeout                         = timeout,  
        gradient_clip_val_range         = gradient_clip_val_range,
        hidden_size_range               = hidden_size_range,
        dropout_range                   = dropout_range,
        learning_rate_range             = learning_rate_range,
        use_learning_rate_finder        = use_learning_rate_finder,
        trainer_kwargs                  = trainer_kwargs,
        log_dir                         = "tb_logs",
        study                           = study_var,
        verbose                         = verbose, # 0: get_verbosity(), 1, warnings, 2. info, 3. debugging
        pruner                          = pruner,
        **gru_configuration)
    
    neptune_logger_wrapper(architecture = 'GRU', study= study, data_module=data_module,configuration=gru_configuration, timeseries_kwargs=timeseries_kwargs)
    return study

def multiple_runs(function_call, n:int):
    """
    @function_call: e
    """
    for i in range(n):
        print(colored("> Starting new round of random search:" + str(i+1) + " of " + str(n), "green"))
        function_call()


def choice(architecture = 'TFT'):
    if architecture == 'TFT': main_tft()
    if architecture == 'LSTM': main_lstm()
    if architecture == 'GRU': main_gru()

    
if __name__ == "__main__":
    #multiple_runs(main_tft, 5)
    multiple_runs(main_lstm, 2)
    multiple_runs(main_gru, 2)
