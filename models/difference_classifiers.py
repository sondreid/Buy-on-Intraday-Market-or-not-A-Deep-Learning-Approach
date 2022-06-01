"""

Network architecture for processing probability that dayahead price exceeds intraday price

"""
from model_definitions import * 
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torchmetrics
from imblearn.metrics import geometric_mean_score            
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



cv = get_cv_type('sliding', 5, **{'val_size':240})


def retrieve_forecasts(folder : str):
    """
    Retrives forecasts from folder containing
    forecast csv files

    """

    current_dir = os.getcwd()
    forecast_files = glob.glob(folder + '/*.csv')
    df = pd.concat(map(pd.read_csv, forecast_files), ignore_index=True)
    df = df.rename(columns = {'Unnamed: 0': 'distance_from_origin'})
    return df

def binary_acc(y_pred, y_test):
    """
    
    """
    
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    return acc




# Expanded variable set


expanded_vars_set = ['DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                    'DK1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                    'GB>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                    'NO1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix', 
                    'NO5>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
                    'NO2 Residual Production Day-Ahead MWh/h H Forecast : prefix']




## Join intraday and dayahead forecasts

intraday_forecasts  = retrieve_forecasts("evaluation_forecasts/" + 'GRU intraday evaluation')
intraday_forecasts = intraday_forecasts.rename(columns = {'forecasts':'intraday_forecasts', 'actuals':'intraday_actuals'})
dayahead_forecasts = retrieve_forecasts("evaluation_forecasts/"+'LSTM dayahead evaluation')
dayahead_forecasts = dayahead_forecasts.rename(columns = {'forecasts':'dayahead_forecasts', 'actuals':'dayahead_actuals'})




difference_df  = dayahead_forecasts.merge(intraday_forecasts, how = 'left', on = 'datetime')
difference_df['intraday_price_difference_forecast'] =  difference_df.dayahead_forecasts - difference_df.intraday_forecasts
difference_df['intraday_price_difference_actuals'] =  difference_df.dayahead_actuals - difference_df.intraday_actuals

difference_df = difference_df.merge(series_df[expanded_vars_set + ['datetime']], how = 'left', on = 'datetime')





def get_actuals(df, date, num_indices : int ):
    """
    Gets the positive or negative index (i.e past or future observations) 
    of historic data
    """
    current_index = df[df['datetime'] == date].index[0]
    before = 0
    after = 0
    if num_indices > 0:
        before = num_indices
    else:
        after = num_indices
    try:
        filtered_df = df.iloc[current_index+before:current_index+after]

    except:
        raise ValueError("> Indices refer to observations too far in the past or future")



def no_skill_ratio(df):
    """

    """
    positive = len(df[(df.intraday_price_difference_actuals > 0)])

    ratio = positive/len(df)
    return ratio





def perform_sigmoid_benchmark(series : pd.Series) -> np.array:
    """
    """

    tensor = torch.Tensor(series.values)
    sigmoid_tensor = torch.sigmoid(tensor)
    return sigmoid_tensor.numpy()
    

def perform_majority_class_benchmark(series: pd.Series) -> list:
    """
    """
    preds = [1 for i in series]
    return preds
    
    




def production_sigmoid():
        """"sumary_line
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        robust_scaler = RobustScaler()
        prod_dayahead_forecasts = retrieve_forecasts("prod_runs/"+"LSTM production")
        prod_dayahead_forecasts = prod_dayahead_forecasts.rename(columns = {'forecasts':'dayahead_forecasts', 'actuals':'dayahead_actuals'})
        prod_intraday_forecasts = retrieve_forecasts("prod_runs/"+"GRU production")[['datetime', 'forecasts']]
        prod_intraday_forecasts  = prod_intraday_forecasts.merge(series_df_evaluation[['datetime', 'intraday_price']], on = 'datetime', how = 'left')
        prod_intraday_forecasts = prod_intraday_forecasts.rename(columns = {'forecasts':'intraday_forecasts', 'intraday_price':'intraday_actuals'})[['datetime', 'intraday_forecasts', 'intraday_actuals']]
        
        prod_forecasts = prod_dayahead_forecasts.merge(prod_intraday_forecasts, how = 'left', on = 'datetime')
        
    

        prod_forecasts['intraday_price_difference_forecasts'] = prod_forecasts.dayahead_forecasts - prod_forecasts.intraday_forecasts

        prod_forecasts['intraday_price_difference_forecasts_scaled'] = robust_scaler.fit_transform(prod_forecasts['intraday_price_difference_forecasts'].to_numpy().reshape(-1, 1))

        ## Append regulation prices
        prod_forecasts = prod_forecasts.merge(series_df_evaluation[['datetime', 'NO2 Price Regulation Down EUR/MWh H Actual : non-forecast', 'NO2 Volume Regulation Net MWh H Actual : non-forecast', 'NO2 Price Regulation Up EUR/MWh H Actual : non-forecast']], on = 'datetime', how = 'left')
        prod_forecasts = prod_forecasts.rename(columns = {'NO2 Price Regulation Down EUR/MWh H Actual : non-forecast'})
        prod_forecasts['intraday_price_difference_classifications'] =perform_sigmoid_benchmark(prod_forecasts.intraday_price_difference_forecasts_scaled)
        
        prod_forecasts.to_csv('difference_probabilities/logit_benchmark.csv')


def retrieve_prod_data():
        """"sumary_line
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        robust_scaler = RobustScaler()
        prod_dayahead_forecasts = retrieve_forecasts("prod_runs/"+"LSTM production")
        prod_dayahead_forecasts = prod_dayahead_forecasts.rename(columns = {'forecasts':'dayahead_forecasts', 'actuals':'dayahead_actuals'})
        prod_intraday_forecasts = retrieve_forecasts("prod_runs/"+"GRU production")[['datetime', 'forecasts']]
        prod_intraday_forecasts  = prod_intraday_forecasts.merge(series_df_evaluation[['datetime', 'intraday_price']], on = 'datetime', how = 'left')
        prod_intraday_forecasts = prod_intraday_forecasts.rename(columns = {'forecasts':'intraday_forecasts', 'intraday_price':'intraday_actuals'})[['datetime', 'intraday_forecasts', 'intraday_actuals']]
        
        prod_forecasts = prod_dayahead_forecasts.merge(prod_intraday_forecasts, how = 'left', on = 'datetime')
        

        prod_forecasts['intraday_price_difference_forecast'] = prod_forecasts.dayahead_forecasts - prod_forecasts.intraday_forecasts

        prod_forecasts['intraday_price_difference_actuals'] = prod_forecasts.dayahead_actuals - prod_forecasts.intraday_actuals

        ## Append regulation prices
        prod_forecasts = prod_forecasts.merge(series_df_evaluation[['datetime', 'NO2 Price Regulation Down EUR/MWh H Actual : non-forecast', 'NO2 Volume Regulation Net MWh H Actual : non-forecast', 'NO2 Price Regulation Up EUR/MWh H Actual : non-forecast']], on = 'datetime', how = 'left')
        prod_forecasts = prod_forecasts.rename(columns = {'NO2 Price Regulation Down EUR/MWh H Actual : non-forecast': 'price_reg_down', 'NO2 Price Regulation Up EUR/MWh H Actual : non-forecast' : 'price_reg_up', 'NO2 Volume Regulation Net MWh H Actual : non-forecast' : 'volume_reg_net'})
        

        # Add expanded variable set
        prod_forecasts = prod_forecasts.merge(series_df_evaluation[expanded_vars_set + ['datetime']], how = 'left', on = 'datetime')
        
        return prod_forecasts

production_df = retrieve_prod_data()   








def cross_entropy_loss(logits,y):
    """
    Cross entropy loss 
    """
    y = y.to(device)
    logits = logits.to(device)
    #bce_loss = nn.BCEWithLogitsLoss()
    loss = F.binary_cross_entropy(logits, y)
    #loss = bce_loss(logits, y)
    return loss



class Convolutional(pl.LightningModule):
    """

    """
    def __init__(self, learning_rate, num_features, hidden_size, num_linear_layers):
        self.save_hyperparameters() # Saving hyperparameters. Ignore unused variable warnings
        super(Convolutional, self).__init__()
        C1 = 256
        C2 = 100
        F1 = 64
        F2 = 64
        self.conv1 = nn.Conv1d(2, 4, kernel_size = 10, stride=2, padding = 5)
        self.fc1 = nn.Linear(260, 1)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.learning_rate)
        return optimizer
    


    def forward(self, batch):
        """
        Used for inference
        """

        x  = torch.reshape(batch, (128, 2))
        print(len(x))
        x = self.conv1(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        """

        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        #y = target.unsqueeze(1)
        loss = cross_entropy_loss(logits, y)
        self.log('loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('accuracy', self.accuracy(logits, y.int()), on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        loss = cross_entropy_loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_accuracy', self.accuracy(logits, y.int()), on_step=False, on_epoch=True, logger=True)
        


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Per default copies forward method
        x, _ = batch
        return self(x)

        
    
    def predict(self, dataloader):
        """
        Iterates through all batches in input dataloader and outputs
        a tensor of predictions
        """
        with torch.no_grad():
                for x, _ in dataloader:
                    # make prediction
                    out = self(x)  
                    #out = move_to_device(out, device="cpu")
                    output.append(out)

        return output


class LSTM_classifier_old(pl.LightningModule):
    """
    LSTM classifier
    """
    def __init__(self, learning_rate, num_features, hidden_size, num_lstm_layers):

        """
        Constructor for MLP
        """
        self.save_hyperparameters() # Saving hyperparameters. Ignore unused variable warnings
        super(LSTM_classifier, self).__init__()

        lstm_layers = [
            nn.LSTM(input_size = num_features, hidden_size = hidden_size, num_layers = num_lstm_layers, batch_first=True, dropout=0.1)]
        
        
        self.output_layer = nn.Linear(hidden_size, 1)

        self.trainaccuracy = torchmetrics.Accuracy()
        self.valaccuracy = torchmetrics.Accuracy()

        self.lstm_sequential = nn.Sequential(*lstm_layers)

    def configure_optimizers(self):
        #TODO
        # Import numerous optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.learning_rate)
        return optimizer
    


    def forward(self, batch):
        """
        Used for inference
        """
        x = batch
        lstm_out, _ = self.lstm_sequential(x)
        x = self.output_layer(lstm_out)
        x = x.view(x.size(0), -1)
        x = torch.relu(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        """

        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        loss = cross_entropy_loss(logits, y)
        self.log('loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('accuracy', self.trainaccuracy(logits, y.int()), on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        val_loss = cross_entropy_loss(logits, y)
        #print("> VAL loss step {}".format(val_loss))
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', self.valaccuracy(logits, y.int()),  prog_bar=True)
  

        
    
    def predict(self, dataloader):
        """
        Iterates through all batches in input dataloader and outputs
        a tensor of predictions
        """
        output = []
        with torch.no_grad():
                for x, _ in dataloader:
                    # make prediction
                    out = self(x)  
                    #out = move_to_device(out, device="cpu")
                    output.append(out)

        return output



class LSTM_classifier(pl.LightningModule):
    """
    MLP classifier


    Loss calculation:
    """
    def __init__(self, model_configuration):

        """
        Constructor for MLP
        """
        self.hparams.update(model_configuration)
        self.save_hyperparameters() # Saving hyperparameters. Ignore unused variable warnings

        lstm_hidden_size =  self.hparams['lstm_hidden_size'] 
        hidden_size =  self.hparams['hidden_size'] 

        
        super(LSTM_classifier, self).__init__()
        self.lstm_layers = nn.LSTM(input_size = self.hparams['num_features'], 
                                hidden_size = lstm_hidden_size, num_layers = self.hparams['num_lstm_layers'], 
                                batch_first=True, dropout=self.hparams['dropout'])
        
        layers = []
        layers_1 = [nn.Linear(self.hparams['num_features'], hidden_size)]
        # hidden layers
        for _ in range(self.hparams['num_linear_layers']):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if self.hparams['batch_normalization']: 
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if self.hparams['dropout'] > 0:
                layers.append(nn.Dropout(self.hparams['dropout']))

        # output layer
        layers_1.append(nn.Linear(hidden_size, 1))
        
        # Initialise training and validation accuracy
        self.trainaccuracy = torchmetrics.Accuracy()
        self.valaccuracy = torchmetrics.Accuracy()

        self.sequential = nn.Sequential(*layers_1)

    def configure_optimizers(self):
        try:
            optimizer_type = self.hparams['optimizer'] 
            optimizer =  optimizer_type(self.parameters(), lr= self.hparams.learning_rate)
        except:
            optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.learning_rate)
        return optimizer
    


    def forward(self, batch):
        """
        Used for inference
        """
        x = batch
        lstm_out, _ = self.lstm_layers(x)
        x = self.sequential(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        """

        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        loss = cross_entropy_loss(logits, y)
        self.log('loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('accuracy', self.trainaccuracy(logits, y.int()), on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        val_loss = cross_entropy_loss(logits, y)
        #print("> VAL loss step {}".format(val_loss))
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', self.valaccuracy(logits, y.int()),  prog_bar=True)
  


        
    
    def predict(self, dataloader, return_index = 0):
        """
        Iterates through all batches in input dataloader and outputs
        a tensor of predictions
        """
        output = []
        with torch.no_grad():
                for x, _ in dataloader:
                    # make prediction
                    out = self(x)  
                    #out = move_to_device(out, device="cpu")
                    output.append(out)

        return output



class MLP_classifier(pl.LightningModule):
    """
    MLP classifier


    Loss calculation:
    """
    def __init__(self, model_configuration):

        """
        Constructor for MLP
        """
        self.hparams.update(model_configuration)
        self.save_hyperparameters() # Saving hyperparameters. Ignore unused variable warnings

        hidden_size =  self.hparams['hidden_size'] 
        
        super(MLP_classifier, self).__init__()
        layers = [nn.Linear(self.hparams['num_features'] , hidden_size), nn.ReLU()]

        # hidden layers
        for _ in range(self.hparams['num_layers']):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if self.hparams['batch_normalization']: 
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if self.hparams['dropout'] > 0:
                layers.append(nn.Dropout(self.hparams['dropout']))
            #layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        # output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        # Initialise training and validation accuracy
        self.trainaccuracy = torchmetrics.Accuracy()
        self.valaccuracy = torchmetrics.Accuracy()

        self.sequential = nn.Sequential(*layers)

    def configure_optimizers(self):
        try:
            optimizer_type = self.hparams['optimizer'] 
            optimizer =  optimizer_type(self.parameters(), lr= self.hparams.learning_rate)
        except:
            optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.learning_rate)
        return optimizer
    


    def forward(self, batch):
        """
        Used for inference
        """
        x = batch
        x = self.sequential(x)
        x = x.view(x.size(0), -1)
        #x = torch.relu(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        """

        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        loss = cross_entropy_loss(logits, y)
        self.log('loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('accuracy', self.trainaccuracy(logits, y.int()), on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        logits = self(x)
        val_loss = cross_entropy_loss(logits, y)
        #print("> VAL loss step {}".format(val_loss))
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', self.valaccuracy(logits, y.int()),  prog_bar=True)
  


        
    
    def predict(self, dataloader, return_index = 0):
        """
        Iterates through all batches in input dataloader and outputs
        a tensor of predictions
        """
        output = []
        with torch.no_grad():
                for x, _ in dataloader:
                    # make prediction
                    out = self(x)  
                    #out = move_to_device(out, device="cpu")
                    output.append(out)

        return output







def perform_predictions_classifier(model, dataloader):
    """
    Performs predictions for a lightning-based classifer 
    """
    
    preds_batch = model.predict(dataloader)
    preds_all = [p.cpu().detach().numpy().flatten() for p in preds_batch]
    preds_array = np.concatenate(preds_all)
    return preds_array




def perform_ensemble_predictions_classifier(trainer, model_call, dataloader):
    """
    Performs ensemble predictions for a lightning-based classifier
    """
    top_model_checkpoint_paths = trainer.checkpoint_callback.best_k_models.keys()
    n_models  = len(top_model_checkpoint_paths)
    preds = pd.DataFrame()
    for model_path in top_model_checkpoint_paths:
        model = model_call.load_from_checkpoint(model_path)
        new_preds = perform_predictions_classifier(model, dataloader)
        try:
            preds  = preds + new_preds
            
        except:
            preds = new_preds
    preds = preds/n_models
    return preds




def classifier_metrics(actuals, preds):
    """
    calculates classification metrics
    """
    
    
    actuals_binary = (actuals > 0).astype(int)
    preds_binary = (preds > 0.5).astype(int)
    test_metrics = {'test_accuracy':          accuracy_score(actuals_binary, preds_binary),
                    'test_gmean'   :        geometric_mean_score(actuals_binary, preds_binary, average = 'binary'),
                    'test_roc_auc_score'    : roc_auc_score(actuals_binary, preds_binary),
                    'test_pr_auc_score'    :  average_precision_score(actuals_binary, preds_binary)}
    
    return test_metrics



def run_classifier_benchmark(df, model, train_indices, val_indices, test_indices, neptune_logger, model_configuration):
    """
    Run benchmark classifiers
    """
    
    features = model_configuration.get('features')
    target = model_configuration.get('target')

    test_df  = df.iloc[test_indices]
    
    sigmoid_benchmark = model(test_df.intraday_price_difference_forecast)
    actuals_and_preds = df.iloc[test_indices]
    actuals_and_preds['intraday_price_difference_classifications'] = sigmoid_benchmark

    
    return actuals_and_preds, {}





def run_classifier_model(df, model, train_indices, val_indices, test_indices, neptune_logger, model_configuration):

    """
    

    Returns forecasts
    """


    batch_size = model_configuration.get('batch_size')
    num_workers = model_configuration.get('num_workers')
    features = model_configuration.get('features')
    target = model_configuration.get('target')

    features_df = df[features + [target]]

    data_module = DifferenceDataModule(features_df, train_indices, val_indices, target, batch_size, num_workers)
    test_df = features_df.iloc[test_indices]
    data_module.set_test_dataloader(test_df)

    
    model_call = model_configuration.get('model_call')

    trainer, model = train_model(data_module, model, neptune_logger, model_configuration)
    if model_configuration.get('use_best_checkpoint'): 
 
            print(colored("> Using best checkpoint....", 'green'))
            best_model_path = trainer.checkpoint_callback.best_model_path
            model = model_call.load_from_checkpoint(best_model_path)
            preds = perform_predictions_classifier(model, data_module.test_dataloader())

    elif model_configuration.get("k_best_checkpoints") is not None:
                print(colored("> Model call {}".format(model_call), 'green'))
                print(colored(" > Using best k best", 'green'))
                preds = perform_ensemble_predictions_classifier(trainer, model_call, data_module.test_dataloader())
            
    else:  preds = perform_predictions_classifier(model, data_module.test_dataloader())
    
    actuals_and_fc = df.iloc[test_indices]
    actuals_and_fc['intraday_price_difference_classifications'] = preds

    train_logged_metrics = dict([ (k,r.item()) for k,r in trainer.logged_metrics.items()]) 
    return actuals_and_fc, train_logged_metrics
    

"""
model_configuration.update(mlp_config)

actuals_and_preds, metrics = run_classifier_model(difference_df, mlp_model, np.arange(0, 1400).tolist(), np.arange(1400, 1500).tolist(), np.arange(1500, 2280).tolist(), None, model_configuration)

# Benchmark run
actuals_and_preds_benchmark, _ =  run_classifier_benchmark(difference_df, perform_sigmoid_benchmark, np.arange(0, 1400).tolist(), np.arange(1400, 1500).tolist(), np.arange(1500, 2280).tolist(), None, {})


# Calculate accuracy

classifier_metrics(actuals_and_preds_benchmark.intraday_price_difference_actuals, actuals_and_preds_benchmark.intraday_price_difference_classifications)
classifier_metrics(actuals_and_preds.intraday_price_difference_actuals, actuals_and_preds.intraday_price_difference_classifications)

"""


"""

model_configuration = {
                'learning_rate': 0.1,  
                'num_features' :3, 
                'hidden_size':128,
                'batch_normalization':True,
                'dropout':0.3,
                'num_layers': 4,
                'optimizer' : torch.optim.RAdam,
                'val_metric':                         'val_loss', 
                'patience':                             30,
                'model_name':                           'MLP model',
                'max_epochs':                           3,
                'accumulate_grad_batches':              2, 
                'neptune_logging': False,
                'use_best_checkpoint':                  True, 
                'model_call':                     MLP_classifier, 
                'k_best_checkpoints' :                   None,
                'gradient_clip_val':                    0.6,
                'min_delta':                             0.001,
                'patience_mode'                      : 'min',
                'batch_size'                         : 128,
                'num_workers' :                        6,
                'target':                     'intraday_price_difference_actuals',
                'features':                     ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] + [],
                'swa_epoch_start':                      20 }

mlp_model = MLP_classifier(model_configuration)"""









def cross_validation_classifications(df: pd.DataFrame, 
                     model,
                     classifier_call,  
                     cv,
                     model_configuration: dict,        
                     verbose = False,
                     gpu_bool = gpus) -> tuple:
        """


        """
        start_cv = time.time()

        # Neptune logger initialisations
        if model_configuration.get("neptune_logger") is False:
                neptune_logging = False
        else: neptune_logging = True
        if neptune_logging:

            neptune_logger = NeptuneLogger(
                api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
                project         = "MasterThesis/Difference",
                source_files=["model_definitions.py", "run_lstm.py", "run_tft.py", "run_deepar.py", "run_tft.py", "difference_classifiers.py", "run_gru.py"]
            )
        else: neptune_logger = None

        model_name  = model_configuration.get('model_name')
        neptune_logger.experiment['model_name'].log(model_name)
        try:
            neptune_logger.log_hyperparams(model.hparams)
            
            #neptune_logger['model/hparams'] = model.hparams
        except:
            print(">No hparams")



        if not os.path.exists("difference_probabilities/cv_runs"):
                os.makedirs("difference_probabilities/cv_runs")
        features = model_configuration.get('features')
        i = 1

        # Store all validaiton metrics


        test_metrics =  {'test_accuracy':    [],
                        'test_gmean'   :     [],
                        'test_roc_auc_score'    : [],
                        'test_pr_auc_score'    :  []}
        for train_indices, val_indices, test_indices in cv.split(df):

            
            fc_and_actuals, train_metrics = classifier_call(df, model, 
                    train_indices, val_indices, test_indices, neptune_logger, model_configuration)

            
            if model_configuration.get('features') is not None:
                features = model_configuration.get('features')
            else: features = []

            fc_and_actuals = fc_and_actuals[['datetime', 'dayahead_actuals', 'intraday_actuals', 'dayahead_forecasts', 'intraday_forecasts',
                     'intraday_price_difference_forecast','intraday_price_difference_actuals', 'intraday_price_difference_classifications'] + features]
            

            fold_test_metrics = classifier_metrics(fc_and_actuals.intraday_price_difference_actuals, fc_and_actuals.intraday_price_difference_classifications)
            fc_and_actuals.to_csv("difference_probabilities/cv_runs/{}_fold{}".format(model_name, i))
            neptune_logger.experiment['fold_data/preds_and_actuals/fold_{}/'.format(i)].upload("difference_probabilities/cv_runs/{}_fold{}".format(model_name, i))
            neptune_logger.experiment['fold_data/test_metrics/test_metrics_fold_{}'.format(i)] = fold_test_metrics
            neptune_logger.experiment['fold_data/train_metrics/train_metrics{}'.format(i)] = train_metrics

            test_metrics['test_accuracy'].append(fold_test_metrics['test_accuracy'])
            test_metrics['test_gmean'].append(fold_test_metrics['test_gmean'])
            test_metrics['test_roc_auc_score'].append(fold_test_metrics['test_roc_auc_score'])
            test_metrics['test_pr_auc_score'].append(fold_test_metrics['test_pr_auc_score'])

            i = i + 1

    

        test_metrics['test_accuracy'] = mean(test_metrics['test_accuracy'])
        test_metrics['test_gmean'] = mean(test_metrics['test_gmean'])
        test_metrics['test_roc_auc_score']=mean(test_metrics['test_roc_auc_score'])
        test_metrics['test_pr_auc_score']=mean(test_metrics['test_pr_auc_score'])

        neptune_logger.experiment['total_performance'] = test_metrics
        neptune_logger.experiment.stop()



def grid_search_classifier(df, model_call, classifier_call, cv, model_configurations):
    """
    Performs grid search for classifier models
    """
    
    keys, values = zip(*model_configurations.items())
    model_configurations_list = [dict(zip(keys, v)) for v in itertools.product(*values)] # All possible model configurations
    print(colored("> Starting grid search. Number of configurations; " + str(len(model_configurations_list)), "blue"))
    print(colored("> All models" + str(model_configurations_list), "blue"))
    for model_configuration in model_configurations_list:
            print("Model config", model_configuration)
            model = model_call(model_configuration)
            print(colored("> Starting new model using config : " + str(model_configuration), 'green'))
            cross_validation_classifications(df, model,classifier_call, cv,  model_configuration)
    print(colored("> Grid search complete", 'green'))



def mlp_run_small_variable_set():

        minimal_variable_set = ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] 

        model_configurations = {
                    'learning_rate':                        [0.05, 0.001],  
                    'num_features' :                       [3], 
                    'hidden_size':                      [256, 128, 512, 16],
                    'batch_normalization':             [True, False],
                    'dropout':                            [0.1, 0.2, 0.4],
                    'num_layers':                           [2,3,5],
                    'optimizer' :                         [torch.optim.RAdam],
                    'val_metric':                         ['val_loss'], 
                    'patience':                             [30],
                    'model_name':                           ['MLP model'],
                    'max_epochs':                           [100],
                    'accumulate_grad_batches':              [3], 
                    'neptune_logging':                       [True],
                    'use_best_checkpoint':                  [False], 
                    'model_call':                     [MLP_classifier], 
                    'k_best_checkpoints' :                   [1,2,5],
                    'gradient_clip_val':                    [0.1, 0.6],
                    'min_delta':                             [0.001],
                    'patience_mode'                      : ['min'],
                    'batch_size'                         : [32],
                    'num_workers' :                        [8],
                    'target':                     ['intraday_price_difference_actuals'],
                    'features':                    [ minimal_variable_set],
                    'swa_epoch_start':                      [20] }


        grid_search_classifier(difference_df, MLP_classifier, run_classifier_model, cv, model_configurations)


def lstm_run_small_variable_set():
        
        
        minimal_variable_set = ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] 

        model_configurations = {
                    'learning_rate':                        [0.01, 0.1,  0.05], 
                    'num_features' :                       [3], 
                    'hidden_size':                      [16, 32, 64],
                    'lstm_hidden_size':                  [64, 128],
                    'batch_normalization':             [True, False],
                    'dropout':                            [0.1, 0.2, 0.4],
                    'num_linear_layers':                           [2],
                    'num_lstm_layers':                         [2],
                    'optimizer' :                         [torch.optim.RAdam],
                    'val_metric':                         ['val_loss'], 
                    'patience':                             [30],
                    'model_name':                           ['LSTM model'],
                    'max_epochs':                           [100],
                    'accumulate_grad_batches':              [2, 5],  
                    'neptune_logging':                       [True],
                    'use_best_checkpoint':                  [False], 
                    'model_call':                     [LSTM_classifier], 
                    'k_best_checkpoints' :                   [1,2,5],
                    'gradient_clip_val':                    [0.1, 0.6],
                    'min_delta':                             [0.001],
                    'patience_mode'                      : ['min'],
                    'batch_size'                         : [32],
                    'num_workers' :                        [8],
                    'target':                     ['intraday_price_difference_actuals'],
                    'features':                    [ minimal_variable_set],
                    'swa_epoch_start':                      [20] }


        grid_search_classifier(difference_df, LSTM_classifier, run_classifier_model, cv, model_configurations)


def lstm_run_extended_variable_set():
        
        
        minimal_variable_set = ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] 
        num_features = len(minimal_variable_set+expanded_vars_set)

        model_configurations = {
                    'learning_rate':                       [0.01, 0.1,  0.05],  
                    'num_features' :                       [num_features], 
                    'hidden_size':                      [16, 32, 64],
                    'lstm_hidden_size':                  [64, 128],
                    'batch_normalization':             [True, False],
                    'dropout':                            [0.1, 0.4],
                    'num_linear_layers':                     [1,3],
                    'num_lstm_layers':                         [2, 4],
                    'optimizer' :                         [torch.optim.RAdam],
                    'val_metric':                         ['val_loss'], 
                    'patience':                             [30],
                    'model_name':                           ['LSTM model'],
                    'max_epochs':                           [100],
                    'accumulate_grad_batches':              [2, 5], 
                    'neptune_logging':                       [True],
                    'use_best_checkpoint':                  [False], 
                    'model_call':                     [LSTM_classifier], 
                    'k_best_checkpoints' :                   [2,5],
                    'gradient_clip_val':                    [0.1, 0.6],
                    'min_delta':                             [0.001],
                    'patience_mode'                      : ['min'],
                    'batch_size'                         : [32],
                    'num_workers' :                        [8],
                    'target':                     ['intraday_price_difference_actuals'],
                    'features':                    [ minimal_variable_set+expanded_vars_set],
                    'swa_epoch_start':                      [20] }


        grid_search_classifier(difference_df, LSTM_classifier, run_classifier_model, cv, model_configurations)




def mlp_run_extended_variable_set():



         # Using expanded variable set
        minimal_variable_set = ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] 
        num_features = len(minimal_variable_set+expanded_vars_set)
        model_configurations = {
                    'learning_rate':                        [0.1, 0.005, 0.05],  
                    'num_features' :                       [num_features], 
                    'hidden_size':                      [64, 32, 128, 256],
                    'batch_normalization':             [True, False],
                    'dropout':                            [0.1, 0.2, 0.4],
                    'num_layers':                           [2,3,5, 7],
                    'optimizer' :                         [torch.optim.RAdam],
                    'val_metric':                         ['val_loss'], 
                    'patience':                             [30],
                    'model_name':                           ['MLP model'],
                    'max_epochs':                           [100],
                    'accumulate_grad_batches':              [2, 4], 
                    'neptune_logging':                       [True],
                    'use_best_checkpoint':                  [False], 
                    'model_call':                     [MLP_classifier], 
                    'k_best_checkpoints' :                   [1,2,5],
                    'gradient_clip_val':                    [0.4],
                    'min_delta':                             [0.001],
                    'patience_mode'                      : ['min'],
                    'batch_size'                         : [64],
                    'num_workers' :                        [8],
                    'target':                     ['intraday_price_difference_actuals'],
                    'features':                    [ minimal_variable_set+expanded_vars_set],
                    'swa_epoch_start':                      [20] }


        grid_search_classifier(difference_df, MLP_classifier, run_classifier_model, cv, model_configurations)


minimal_variable_set = ['intraday_forecasts', 'intraday_price_difference_forecast', 'dayahead_forecasts'] 
num_features = len(minimal_variable_set+expanded_vars_set)
model_configuration = {
                    'learning_rate':                        0.1,  
                    'num_features' :                     num_features, 
                    'hidden_size':                      64,
                    'batch_normalization':             True,
                    'dropout':                            0.1,
                    'num_layers':                         2,
                    'optimizer' :                         torch.optim.RAdam,
                    'val_metric':                         'val_loss', 
                    'patience':                             30,
                    'model_name':                           'MLP model',
                    'max_epochs':                           100,
                    'accumulate_grad_batches':              4, 
                    'neptune_logging':                       True,
                    'use_best_checkpoint':                  False, 
                    'model_call':                     MLP_classifier, 
                    'k_best_checkpoints' :                   5,
                    'gradient_clip_val':                    0.4,
                    'min_delta':                             0.001,
                    'patience_mode'                      : 'min',
                    'batch_size'                         : 64,
                    'num_workers' :                        8,
                    'target':                     'intraday_price_difference_actuals',
                    'features':                    minimal_variable_set+expanded_vars_set,
                    'swa_epoch_start':                      20 
                    }



def refit_classifier(df, model_configuration):



    batch_size = model_configuration.get('batch_size')
    num_workers = model_configuration.get('num_workers')
    features = model_configuration.get('features')
    target = model_configuration.get('target')

    features_df = df[features + [target]]

    train_indices, val_indices = train_test_split(df.index.values, test_size = 150, shuffle = False)

    data_module = DifferenceDataModule(features_df, train_indices, val_indices, target, batch_size, num_workers)

    model_call = model_configuration.get('model_call')
    model = model_call(model_configuration)

    trainer, model = train_model(data_module, model, None, model_configuration)


    return trainer, model, data_module


def run_production_model(model_configuration, test_df):
    
    """
    

    """
  
    trainer, model, data_module = refit_classifier(difference_df, model_configuration )
    features = model_configuration.get('features')
    target = model_configuration.get('target')
    
    features_test_df = test_df[features + [target]]


    data_module.set_test_dataloader(features_test_df)
    
    preds = perform_predictions_classifier(model, data_module.test_dataloader()) 
    test_df['difference_probabilities'] = preds

    test_df = test_df[['datetime', 'intraday_forecasts', 'intraday_price_difference_forecast','dayahead_forecasts', 'intraday_actuals', 'dayahead_actuals', 'intraday_price_difference_actuals', 'volume_reg_net', 'price_reg_down', 'price_reg_up', 'difference_probabilities']]
    test_df.to_csv('difference_probabilities/{}_prod_probabilities.csv'.format(model_configuration.get('model_name')))

    return test_df









if __name__ == '__main__':

    # Run benchmarks

    #####################################
    #cross_validation_classifications(df = difference_df, model =  perform_sigmoid_benchmark, classifier_call =run_classifier_benchmark, cv = cv, model_configuration = {'model_name': 'Benchmark classifier'})

    #cross_validation_classifications(df = difference_df, model =  perform_majority_class_benchmark, classifier_call =run_classifier_benchmark, cv = cv, model_configuration = {'model_name': 'Majority class benchmark'})
    run_production_model(model_configuration=model_configuration, test_df = production_df)

    #lstm_run_extended_variable_set()
    #lstm_run_small_variable_set()
    #run_small_variable_set()
    #lstm_run_small_variable_set()
    #mlp_run_extended_variable_set()