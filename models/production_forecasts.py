"""
Script for generating production forecasts



"""
#Load dependencies
from model_definitions import * 
from run_lstm import * 
from performance_evaluation import * 

## Optimal models


df_production = pd.concat([series_df, series_df_evaluation])



prediction_length  = 38



## LSTM dayahead

def setup_prod_lstm():

    target = 'dayahead_price'

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
    return lstm_model, lstm_model_config, timeseries_kwargs

def setup_prod_gru():
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
    return gru_model, gru_model_config, timeseries_kwargs







### Initiallise models and configurations 

lstm_model, lstm_model_config, lstm_timeseries_kwargs = setup_prod_lstm()
gru_model, gru_model_config, gru_timeseries_kwargs = setup_prod_gru()







def production_forecasts(df, model, model_call, model_configuration, production_config, timeseries_kwargs):
    """

    Production config:

        starting_index: The index of the first observation denoting the start of the test period
        num_prod_days: Total days of production 
        starting_day: n-days from the starting index (24*n hours) to start production ** defaults to 0
    """

    try:
        model_name = production_config.get('model_name')
    except:
        raise ValueError("> Missing model name")
    # Neptune logger initialisations
    if model_configuration.get("neptune_logger") is False:
            neptune_logging = False
    else: neptune_logging = True
    if neptune_logging:
        neptune_logger = neptune.init(
            project="MasterThesis/production",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            name = production_config.get('model_name'),
            description = "Production forecasts",
            source_files=["model_definitions.py", "run_lstm.py", "production_forecasts.py", "run_deepar.py", "performance_evaluation.py", "run_tft.py", "run_gru.py"]
        )  
        neptune_logger['Type']                                  = "Cross validated run"
        
        neptune_logger['Target']                                = model.target_names[0]
        neptune_logger['Architecture']                          = production_config.get('model_name')
        neptune_logger['model/configuration']                   = model_configuration
        neptune_logger['model/timeseries']                      = timeseries_kwargs
        neptune_logger['model/prod_config']                     = production_config
        
        try:
            neptune_logger['model/hparams'] = model.hparams
        except:
            print(">No hparams")

    else: neptune_logger = None


    if not os.path.exists("prod_runs"):
         os.makedirs("prod_runs")

    production_start_index = production_config.get('starting_index')
    num_prod_days = production_config.get('num_prod_days')

    starting_day = 0 
    if production_config.get('starting_day') is not None:
        starting_day = production_config.get('starting_day')

    for  i in range(starting_day, num_prod_days):
        print(colored("> Day {} of {} ".format(i+1, num_prod_days), 'cyan'))


        days_forward = i *24
        train_and_val_indices = df.index.values[0:production_start_index+days_forward]

        
        train_indices, val_indices = train_test_split(train_and_val_indices, test_size = 114, shuffle = False)
        
        train_indices = train_indices.tolist() + val_indices.tolist()

        test_indices = np.arange(production_start_index+days_forward, production_start_index+days_forward+38)
        print(colored("> Train indices first {} last {} val indices first {} last {} test indices first {} last {}".format(
            train_indices[0], train_indices[-1], val_indices[0], val_indices[-1], test_indices[0], test_indices[-1]), 'blue'))

        actuals_and_fc, _ = model_call(df, model, model_configuration, timeseries_kwargs, train_indices, val_indices.tolist(), test_indices.tolist() )
         
        print(colored("> Len actuals and fc {}".format(len(actuals_and_fc), 'green')))
         
        last_24_hour_forecasts = actuals_and_fc.tail(24)

        last_24_hour_forecasts.to_csv('prod_runs/{}/production_day_{}_forecasts.csv'.format(model_name, i+1))

        if neptune_logging: neptune_logger['forecasts/{}_day_{}'.format(model_name, i+1)].upload('prod_runs/{}/production_day_{}_forecasts.csv'.format(model_name, i+1))
    current_dir = os.getcwd()
    forecast_files = glob.glob('prod_runs/*.csv')
    production_forecasts = pd.concat(map(pd.read_csv, forecast_files), ignore_index=True)
    production_forecasts = df.rename(columns = {'Unnamed: 0': 'distance_from_origin'})
    if neptune_logging: neptune_logger['all_forecasts'].upload(production_forecasts)
    return production_forecasts
        



    



if __name__ == '__main__':
    production_config =  {'model_name': 'LSTM production', 'num_prod_days':40, 'starting_day': 16, 'starting_index':24338}
    production_forecasts(df_production, lstm_model, evaluation_pytorch_models, model_configuration = lstm_model_config, production_config = production_config, timeseries_kwargs=lstm_timeseries_kwargs)
    
    #production_config =  {'model_name': 'GRU production', 'num_prod_days':40, 'starting_index':24338}
    #production_forecasts(df_production, gru_model, evaluation_pytorch_models, model_configuration = gru_model_config, production_config = production_config, timeseries_kwargs=gru_timeseries_kwargs)

