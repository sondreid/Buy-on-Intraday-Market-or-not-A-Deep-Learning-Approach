


from typing import List, Tuple
import torch
torch.cuda.is_available()


from model_definitions import * 
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models import AutoRegressiveBaseModel, BaseModel
from pytorch_forecasting import LSTM, AutoRegressiveBaseModelWithCovariates, TimeSeriesDataSet, TemporalFusionTransformer, metrics


# Using historic data as a a test
intraday_price_difference = series_df_shifted[['intraday_price']].copy()
data_module_univariate = DifferenceDataModule(df = intraday_price_difference, target = 'intraday_price', batch_size=256, num_workers=4)

class LSTMModel(pl.LightningModule):
    """
    Simple MLP classifier

    Loss calculation:
    """
    def __init__(self, learning_rate):
        self.save_hyperparameters() # Saving hyperparameters. Ignore unused variable warnings
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size = 1, hidden_size = 32, num_layers = 1, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(32, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.learning_rate)
        return optimizer

    def forward(self, batch):
        x = batch
        lstm_out, _ = self.lstm(x)
        preds = self.linear(lstm_out[:,-1])
        print("> Preds")
        return preds


    def training_step(self, batch, batch_idx):
        x,y = batch
        #x = x.view(x.size(0, -1))
        preds = self(x)
        loss = F.mse_loss(preds, x)
        return loss




timeseries_kwargs = {'time_idx': 'seq',
                     'target':'intraday_price', 
                     'time_varying_unknown_reals':          ['intraday_price'],
                      'time_varying_known_reals': [],
                     'time_varying_unknown_categoricals':   [], 
                     'time_varying_known_categoricals':   [], 
                     'batch_size' : 516,
                     'num_workers' : 8,
                     'scale': False,
                     'min_encoder_length':10,
                     'max_encoder_length':10,
                     'min_prediction_length':2,
                     'max_prediction_length':30,
                     'group_ids': ['time_groups'],
                     
                     'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True)}



data_module = DataModuleTimeSeries(
            df                                      = series_df_shifted, 
            **timeseries_kwargs)


model_kwargs = {'loss': metrics.MAE()}

lstm_model = LSTMModel(learning_rate=0.1)
lstm_model.summarize()
lstm_model.hparams

callbacks = EarlyStopping(patience = 10, monitor='val_loss')

trainer = pl.Trainer(
        max_epochs                  = 50,
        callbacks= [callbacks],
        gpus                        = 0,
        check_val_every_n_epoch=1,
        strategy = None
    )


#trainer.fit(lstm_model,  data_set.to_dataloader(train = True, shuffle = False, batch_size=20))
trainer.fit(lstm_model, datamodule = data_module)
data_module.set_test_dataloader(series_df_evaluation.iloc[0:100])








class LSTMDecoderEncoder(AutoRegressiveBaseModel):

    """
    LSTTM with 
    """
    def __init__(
        self,
        target: str,
        target_lags: Dict[str, Dict[str, int]],
        n_layers: int,
        hidden_size: int,
        model_configuration,
        dropout: float = 0,
        **kwargs,
    ):
        # hpparams saves hyperparameters
        self.save_hyperparameters()
        
        # calculate the size of all concatenated embeddings + continous variables
        """        n_features = sum(
                    embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
                ) + + len(self.reals)"""
        self.encoded_cont = len(timeseries_kwargs.get('time_varying_unknown_reals'))
        self.n_features = self.encoded_cont + \
        len(timeseries_kwargs.get('time_varying_unknown_categoricals'))+ 1
        
        print("> N features", self.n_features)
        n_features = 2

        super().__init__(**kwargs)
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size= self.encoded_cont,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hparams.hidden_size, 1)

    def encode(self, x: Dict[str, torch.Tensor]):
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims= 2
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging

        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        _, hidden_state = self.lstm(
            input_vector, lengths=effective_encoder_lengths, enforce_sorted=False  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        
        #print(colored("> Encode complete", 'green'))
        
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=2
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)

            # transform into right shape
            prediction = self.output_layer(lstm_output)
            prediction = self.transform_output(prediction, target_scale=x["target_scale"])

            # predictions are not yet rescaled
            return prediction

        else:  # prediction mode
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                # overwrite at target positions
                x[:, 0, target_pos] = lagged_targets[-1]  # take most recent target (i.e. lag=1)
                lstm_output, hidden_state = self.lstm(x, hidden_state)
                # transform into right shape
                prediction = self.output_layer(lstm_output)[:, 0]  # take first timestep
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )

            # predictions are already rescaled
            return output

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state
        
        # Add support for categorical variables
        return self.to_network_output(prediction=output)





timeseries_kwargs = {'time_idx': 'seq',
                     'target':'intraday_price', 
                     'time_varying_unknown_reals':          time_varying_known_reals,
                      'time_varying_known_reals': [],
                     'time_varying_unknown_categoricals':   time_varying_unknown_categoricals, 
                     'time_varying_known_categoricals':   [], 
                     'static_categoricals':                 ['time_groups'],
                        'categorical_encoders':  { 
                                            'covid':        NaNLabelEncoder(add_nan = True)
                                            },
                     'batch_size' : 516,
                     'num_workers' : 8,
                     'scale': False,
                     'min_encoder_length':10,
                     'max_encoder_length':10,
                     'min_prediction_length':2,
                     'max_prediction_length':30,
                     'group_ids': ['time_groups'],
                     
                     'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True)}



data_module = DataModuleTimeSeries(
            df                                      = series_df_shifted, 
            **timeseries_kwargs)


model_kwargs = {'loss': metrics.MAE()}

lstm_model = LSTMDecoderEncoder.from_dataset(data_module.train_dataloader().dataset, n_layers = 1, hidden_size = 30, model_configuration = timeseries_kwargs, **model_kwargs)
lstm_model.summarize()
lstm_model.hparams

callbacks = EarlyStopping(patience = 10, monitor='val_loss')

trainer = pl.Trainer(
        max_epochs                  = 50,
        callbacks= [callbacks],
        gpus                        = 0,
        check_val_every_n_epoch=1,
        strategy = None
    )


#trainer.fit(lstm_model,  data_set.to_dataloader(train = True, shuffle = False, batch_size=20))
trainer.fit(lstm_model, datamodule = data_module)
data_module.set_test_dataloader(series_df_evaluation.iloc[0:100])

trainer.test(dataloaders=data_module.test_dataloader())

#lstm_model.predict(data_module.test_dataloader())

plot_predictions_simple(lstm_model, data_module.test_dataloader())

trainer.predict(lstm_model, dataloaders=data_module.test_dataloader())


