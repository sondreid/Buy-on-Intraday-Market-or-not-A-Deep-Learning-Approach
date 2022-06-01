from model_definitions import * 
from sklearn.metrics import accuracy_score
import torchmetrics

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




series_expanded_intraday = series_df_shifted[['intraday_price_difference', 'intraday_price', 'dayahead_price']].copy()
series_expanded_intraday_evaluation = series_df_evaluation[['intraday_price_difference', 'intraday_price', 'dayahead_price']].copy()

expanded_data_set = PandasDataset(series_expanded_intraday, 'intraday_price_difference')


batch_size = 128

kwargs = {'drop_last':True}
data_module_conv = DifferenceDataModule(df = series_expanded_intraday, target = 'intraday_price_difference', batch_size=128, num_workers=8, **kwargs)






test_tensor = torch.randn(3, 2, 4)

m = nn.Conv1d(128, 4, 2, stride=2)
output = []
x = None
with torch.no_grad():
        for x, _ in data_module_multivariate.train_dataloader():
            # make prediction
            print(">len {}".format(len(x)))
            x = x
            break
            print("> X before unsqueeze{}".format(x))
            x = torch.flatten(x)
            print("> X before unsqueeze{}".format(x))



# Using batch size 128

m = nn.Conv1d(128, 4, 2, stride=2)
output = []
with torch.no_grad():
        for x, _ in data_module_multivariate.train_dataloader():
            # make prediction
            print(">len {}".format(len(x)))
            if len(x) != 128:
                continue
            out = m(x)  
            #out = move_to_device(out, device="cpu")
            output.append(out)
output




# Checking lengths



with torch.no_grad():
        for x, _ in data_module_multivariate.train_dataloader():
            # make prediction

            
            print(">len {}".format(len(x)))
            """           if len(x) != 128:
                            continue"""






# Trying to reshape

m = nn.Conv1d(2, 4, kernel_size = 10, stride=2, padding = 5)
output = []
with torch.no_grad():
        for x, _ in data_module_conv.train_dataloader():
            # make prediction

            
            #print(">len {}".format(len(x)))
            """      if len(x) != 128:
                                                            continue"""
            #adaptive_pooling = nn.AdaptiveMaxPool1d(20)
            #x = adaptive_pooling(x)
            x = torch.reshape(x, (2, len(x)))
            out = m(x)  
            #out = move_to_device(out, device="cpu")
            output.append(out)
output




linear_layer = nn.Linear(260, 1)
linearized_outputs = []
for o in output:
    
    o = torch.flatten(o)
    print(len(o))
    linearized_output = linear_layer(o)
    linearized_outputs.append(linearized_output)