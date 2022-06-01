"""
Script for generating learning curves


"""



from analysis import *
import sys

#sys.path.insert(0,'/models/')
try:
    cwd = os.getcwd()
    os.chdir("models")
except:
    print("> Already in models directory")
from model_definitions import * 
from performance_evaluation import *



#######################################################################
#######################################################################
############################ Refitting models ########################



# LSTM dayahead
target = 'dayahead_price'
prediction_length = 38
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
lstm_data_module,  lstm_model, filtered_params = setup_lstm(series_df_shifted, lstm_model_config, timeseries_kwargs)
lstm_data_module.set_val_dataloader(series_df_evaluation.iloc[0:100])

kwargs = {'model_name':'lstm_learning_curve'}
trainer, model = train_model(
        data_module = lstm_data_module, 
        model       = lstm_model,  
        gpu_bool    = gpus, 
        neptune_logger = None,
        model_configuration = filtered_params, **kwargs)




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
gru_data_module, gru_model, filtered_params = setup_lstm(series_df_shifted, gru_model_config, timeseries_kwargs)
gru_data_module.set_val_dataloader(series_df_evaluation.iloc[0:100])

kwargs = {'model_name':'gru_learning_curve'}
trainer, model = train_model(
        data_module = gru_data_module, 
        model       = gru_model,  
        gpu_bool    = gpus, 
        neptune_logger = None,
        model_configuration = filtered_params, **kwargs)


#######################################################################
#######################################################################
############################ Drawing  learning curves #################





def plot_learning_curves(model_name,  title):
    """
    Plots a learning curve
    """
    
    sns_theme = 'white'
    sns_palette = 'Set2'
    font = 'Computer Modern'
    sns.set(rc={'figure.figsize':(75,35)})

    val_curve =pd.read_csv("../images/learning_curves/{}/val.csv".format(model_name))
    val_curve['type'] = 'Validation'

    train_curve =pd.read_csv("../images/learning_curves/{}/train.csv".format(model_name))
    train_curve['type'] = 'Training'


    sns.set_context('paper', font_scale = 7)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    sns.despine(offset = 0)
    #plt.legend(loc=legend_loc)
    fig, ax =plt.subplots(1,2)

    pl1 = sns.lineplot(
        data        = val_curve, 
        x           = val_curve.Step, 
        y           = val_curve.Value, 
        ci          = None,
        ax = ax[0],
        lw          = 3, 
        palette     = sns_palette)

    pl1.legend(title = 'Validation', loc='upper left')
    pl1.set(xlabel='Step', ylabel='MAE')


    pl2 = sns.lineplot(
        data        = train_curve, 
        x           = train_curve.Step, 
        y           = train_curve.Value, 
        ci          = None,
        ax = ax[1],
        lw          = 3, 
        palette     = sns_palette)

    pl2.legend(title = 'Training', loc='upper left')
    pl2.set(xlabel='Step', ylabel='MAE')


    pl1.xaxis.set_major_formatter(ticker.ScalarFormatter())


    plt.savefig('../images/learning_curves/' + title + '.png', bbox_inches='tight', edgecolor='none', facecolor="white")
    plt.show()




plot_learning_curves('LSTM', 'LSTM_lc')
