from model_definitions import *
from run_lstm import * 
from run_deepar import * 

"""

"""


target = 'dayahead_price'



timeseries_kwargs = {
        'time_idx':                            'seq',
        'target':                              target, 
        'min_encoder_length' :                 1,  
        'max_encoder_length' :                 168,  
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
        'batch_size':                           128,
        'num_workers':                           7,
        'scale':                                False} 






lstm_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [7],  
        'max_epochs':                           [70], 
        'patience':                             [15],
        'cell_type':                            ['LSTM'],
        'hidden_size':                          [128],
        'rnn_layers':                           [2],
        'loss':                                 [metrics.MAE()],   
        'dropout':                              [0.3],
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
        'gradient_clip_val'     :                 [0.5],
        #'target_lags':                          {},
        #'logging_metrics':                      None
        'optimizer':                            ['ranger'],
        'learning_rate' :                       [0.001],
        'accumulate_grad_batches':              [3], 
        'use_best_checkpoint':                  [True], 
        'reduce_on_plateau_patience':           [2],
        'reduce_on_plateau_reduction' :         [2.0],
        'model_call' :                         [RecurrentNetwork],
        'neptune_logger':                       [True],
        'swa_epoch_start':                        [12]
        }





lstm_model_config = take_first_model(lstm_model_config)

data_module, lstm_model, filtered_lstm_model_parameters = setup_lstm(series_df_shifted, lstm_model_config, timeseries_kwargs)
# Find initial learning rate
#find_initial_learning_rate(lstm_model, data_module, lstm_model_config)

run_specific_fold(series_df_shifted, lstm_model, cv, 0, 'LSTM_test', timeseries_kwargs,  filtered_lstm_model_parameters, verbose = True)

################## GRU ################

gru_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [7],  
        'max_epochs':                           [70], 
        'patience':                             [17],
        'cell_type':                            ['GRU'],
        'hidden_size':                          [256],
        'rnn_layers':                           [4],
        'loss':                                 [metrics.MAE()],   
        'dropout':                              [0.3],
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
        'gradient_clip_val'                     : [0.6],
        #'target_lags':                          {},
        #'logging_metrics':                      None
        'optimizer':                            ['ranger'],
        'learning_rate' :                       [0.001],
        'accumulate_grad_batches':              [2], 
        'use_best_checkpoint':                  [True], 
        'reduce_on_plateau_patience':           [2],
        'reduce_on_plateau_reduction' :         [2.0],
        'model_call' :                         [RecurrentNetwork],
        'neptune_logger':                       [True],
        'swa_epoch_start':                        [12]
        }


gru_model_config = take_first_model(gru_model_config)

_, gru_model, filtered_gru_model_parameters = setup_lstm(series_df_shifted, gru_model_config, timeseries_kwargs)

run_specific_fold(series_df_shifted, gru_model, cv, 4, 'gru_test', timeseries_kwargs,  filtered_gru_model_parameters, verbose = True)

find_initial_learning_rate(gru_model, data_module, gru_model_config)


## Deep AR ####################

deepar_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [9],  
                'max_epochs':                           [70], 
                'patience':                             [20],
                'cell_type':                            ['LSTM'],
                'hidden_size':                          [64],
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
                'optimizer':                            ['ranger'],
                'gradient_clip_val'                     :[0.2],
                'learning_rate' :                       [0.0001],
                'accumulate_grad_batches':              [2], 
                'use_best_checkpoint':                  [True], 
                'reduce_on_plateau_patience':           [3],
                'reduce_on_plateau_reduction' :         [3.0],
                'model_call' :                         [DeepAR],
                'neptune_logger':                       [True],
                'swa_epoch_start' :                    [16] 
        }


deepar_model_config = take_first_model(deepar_model_config)

_, deepar_model, filtered_deepar_model_parameters = setup_deepar(series_df_shifted, deepar_model_config, timeseries_kwargs)

run_specific_fold(series_df_shifted, deepar_model, cv, 4, 'DeepAR_test', timeseries_kwargs,  filtered_deepar_model_parameters, verbose = True)



### TFT ####
from run_tft import * 

timeseries_kwargs = {
        'time_idx':                            'seq',
        'target':                              target, 
        'min_encoder_length' :                 1,  
        'max_encoder_length' :                 80,  
        'group_ids':                           ['time_groups'],
        'min_prediction_length':               max_prediction_length,
        'max_prediction_length':               max_prediction_length+2,
        'time_varying_known_reals' :           time_varying_known_reals,
        'time_varying_unknown_reals':          time_varying_unknown_reals,
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
        'batch_size':                           128,
        'num_workers':                           11,
        'scale':                                False
        }
tft_model_config =   {
        'batch_size':                           [128], 
        'num_workers':                          [11],  
        'max_epochs':                           [70], 
        'patience':                             [9],
        'hidden_size':                          [16],
        'lstm_layers':                          [1], 
        'dropout':                              [0.1],
        'output_size':                          [1],
        'loss':                                 [metrics.MAE()],                
        'attention_head_size':                  [4],
        'min_delta':                            [0.2],
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
        'hidden_continuous_size':               [4],
        #'hidden_continuous_sizes':              {},
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},    
        'learning_rate':                        [0.0005],
        #'gradient_clip_val':                    [0.00008198106964873125],
        'log_interval':                         [1],
        'log_val_interval':                     [1],
        #'log_gradient_flow':                    False,
        #'monotone_constaints':                  {},
        #'share_single_variable_networks':       False,
        #'logging_metrics':                      None
        #'optimizer_params': [{'lr':0.1}],
        'optimizer':                      ['ranger'],
        'accumulate_grad_batches':               [2], 
        'use_best_checkpoint':                  [True], 
        'reduce_on_plateau_patience':           [3],
        'reduce_on_plateau_reduction' :         [3.0],
        'model_call' :                         [TemporalFusionTransformer],
        'neptune_logger':                       [True],
        'swa_epoch_start' :                    [16] 
        }




tft_model_config = take_first_model(tft_model_config)

_, tft_model, filtered_tft_model_parameters = setup_tft(series_df_shifted, tft_model_config, timeseries_kwargs)

run_specific_fold(series_df_shifted, tft_model, cv, 4, 'TFT_test', timeseries_kwargs,  filtered_tft_model_parameters, verbose = True)


