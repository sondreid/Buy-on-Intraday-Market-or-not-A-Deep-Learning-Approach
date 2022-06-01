"""

Model evaluation script

"""
import models
# adding Folder_2 to the system path
#sys.path.insert(0, '/models'/)

from analysis import *
import sys
#sys.path.insert(0,'/models/')
try:
    cwd = os.getcwd()
    os.chdir("models")
except:
    print("> Already in models directory")


from models.run_tft import *
from models.benchmark import * 

timeseries_kwargs = {
    'time_idx':                            'seq',
    'target':                              target, 
    'max_encoder_length' :                 70,  
    'group_ids':                           ['time_groups'],
    'max_prediction_length':               200,
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
    'num_workers':                          8,  
    'max_epochs':                           100, 
    'patience':                             10,
    'hidden_size':                          15,
    'lstm_layers':                          5, 
    'dropout':                              0.28472627877645074,
    'output_size':                         7,
    'loss':                                 QuantileLoss(),                
    'attention_head_size':                  8,
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
    'learning_rate':                        0.02429598212510496,
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
checkpoint = torch.load('refit_runs/tft_refit_1_model_weights.pth')
data_module, tft_model, filtered_tft_model_parameters = setup_tft(series_df_shifted, tft_model_config, timeseries_kwargs)
tft_model.load_state_dict(checkpoint)


trainer = pl.Trainer(
        max_epochs                  = tft_model_config.get('max_epochs'),
        gpus                        = 1,
        accelerator                 = 'gpu',
        benchmark = True, 
        devices = 1,
        gradient_clip_val           = 0.1
    )

trainer.predict(tft_model, data_module.test_dataloader(), ckpt_path='refit_runs/test_trainer.cpkt')



##### Plots of benchmark models 

best_model_order = {'P': 3, 'D': 1, 'Q': 3, 'p': 1, 'd': 1, 'q': 2, 's': 18} 
benchmark_forecasts = arima_model(series_df_shifted.intraday_price_difference, series_df_evaluation.intraday_price_difference, orders = best_model_order)

actuals = remove_nans(series_df_evaluation, 'linear')
preds_and_actuals = pd.DataFrame({'preds': benchmark_forecasts, 'actual': actuals.intraday_price_difference})
preds_and_actuals['hours'] = np.arange(0, len(preds_and_actuals)).tolist()

preds_and_actuals_short = preds_and_actuals.iloc[0:200]

    
sns.set_context('paper', font_scale = 4)
sns.set_style(sns_theme)
sns.set_style({'font.family':'serif', 'font.serif':font})

ax = sns.lineplot(
    data        = preds_and_actuals_short, 
    x           = preds_and_actuals_short.hours, 
    y           = preds_and_actuals_short.preds, 
    palette     = sns_palette)


ax = sns.lineplot(
    data        = preds_and_actuals_short, 
    x           = preds_and_actuals_short.hours, 
    y           = preds_and_actuals_short.actual, 
    palette     = sns_palette)

plot_predictions_simple(tft_model, remove_nans(series_df_evaluation, 'linear'))

data_module.set_test_dataloader(series_df_evaluation)
trainer.predict(tft_model, data_module.test_dataloader(), ckpt_path='refit_runs/test_trainer.cpkt')
