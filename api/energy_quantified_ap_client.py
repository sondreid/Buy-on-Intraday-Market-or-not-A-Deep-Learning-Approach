"""

Data retrevial using Montel's python API

"""
# Import modules
from datetime import datetime, date, timedelta, time
from distutils.errors import DistutilsPlatformError
import itertools
from multiprocessing.sharedctypes import Value
import pandas as pd
from energyquantified import EnergyQuantified
import pickle
from energyquantified.time import Frequency, get_datetime, CET, UTC
from energyquantified.metadata import Aggregation, Filter


api_key =  'a_hidden_key'


# Initialize client
eq = EnergyQuantified(api_key= api_key)



# Specify zones
zone_list_norway = ['NO1', 'NO2','NO3', 'NO4', 'NO5']
zone_list_sweden = ['SE1', 'SE2', 'SE3', 'SE4']
zone_list_denmark = ['DK1', 'DK2']


nordics = zone_list_norway + zone_list_sweden + zone_list_denmark + ['FI']

all_zones = zone_list_norway + zone_list_sweden + \
   zone_list_denmark + ['LT'] + ['FI'] + ['RU'] + ['EE'] + ['LV']


  

def retrieve_timeseries_for_all_zones(search_term_without_country, series_name, zone_lists, frequency, **time_kwargs):
      """
      NOT IN USE
      Retrieves an instance time series for all tags and specified zones
      """

      df = pd.DataFrame(columns = ['value','date', 'zone', 'unit'])
      for zone in zone_lists:
         search_term = zone +  search_term_without_country
         try:  # Check if the search term exists
            tags = eq.instances.tags(search_term)
         except: # Continue if the search term does not exist
               print("> No series found..")
               continue
         for tag in tags:
               search_term_data = eq.instances.relative(
                  search_term,
                  begin = get_datetime(time_kwargs.get('start_year'), time_kwargs.get('start_month'), time_kwargs.get('start_day'), tz = CET),
                  end   = get_datetime(time_kwargs.get('end_year'), time_kwargs.get('end_month'), time_kwargs.get('end_day'), tz = CET),
                  tag = tag,
                  days_ahead = 2,
                  issued = 'latest',
                  frequency=frequency
               ).to_dataframe()
            
               #print('> Search term data', search_term_data)
               search_term_data['date'] = search_term_data.index
               search_term_data = search_term_data.reset_index(drop=True)
               search_term_data['tag'] = tag
               search_term_data['zone'] = zone
               search_term_data.columns =  [series_name, 'date', 'tag', 'zone']
               df = pd.concat([df, search_term_data])

      return df




def retrieve_remit(search_term, series_name, zone, **time_kwargs):
      """
      Retrieves an instance time series for all tags, a single specfiied zone, and a list of 
      frequencies
      """

      df = pd.DataFrame(columns = ['value','date', 'tag', 'zone', 'unit'])
      try:
            
         search_term_data = eq.period_instances.latest(
            search_term,
            begin = get_datetime(time_kwargs.get('start_year'), time_kwargs.get('start_month'), time_kwargs.get('start_day'), tz = CET),
            end   = get_datetime(time_kwargs.get('end_year'), time_kwargs.get('end_month'), time_kwargs.get('end_day'), tz = CET),
         ).to_dataframe(frequency=Frequency.PT1H)
         search_term_data['date'] = search_term_data.index
         search_term_data = search_term_data.reset_index(drop=True)
         search_term_data['tag'] = 'non-forecast'
         search_term_data['zone'] = zone
         search_term_data['unit'] = series_name
         search_term_data.columns =  ['value', 'date',  'tag', 'zone', 'unit']
         return search_term_data
      except:
         print('> Series does not have specified frequency')
         return df





def retrieve_instance(search_term, series_name, zone, frequencies, **time_kwargs):
      """
      Retrieves an instance time series for all tags, a single specfiied zone, and a list of 
      frequencies
      """

      df = pd.DataFrame(columns = ['value','date', 'tag', 'zone', 'unit'])


      for frequency in frequencies:
         try:
               
            search_term_data = eq.timeseries.load(
               search_term,
               begin = get_datetime(time_kwargs.get('start_year'), time_kwargs.get('start_month'), time_kwargs.get('start_day'), tz = CET),
               end   = get_datetime(time_kwargs.get('end_year'), time_kwargs.get('end_month'), time_kwargs.get('end_day'), tz = CET),
               frequency=frequency,
               aggregation=Aggregation.AVERAGE
            ).to_dataframe()
            search_term_data['date'] = search_term_data.index
            search_term_data = search_term_data.reset_index(drop=True)
            search_term_data['tag'] = 'non-forecast'
            search_term_data['zone'] = zone
            search_term_data['unit'] = series_name
            search_term_data.columns =  ['value', 'date',  'tag', 'zone', 'unit']
            return search_term_data
         except:
            print('> Series does not have specified frequency')
            continue


      return df





def retrieve_timeseries(search_term, series_name, zone, frequencies,  **time_kwargs):
      """
      Retrieves an instance time series for all tags, a single specfiied zone, and a list of 
      frequencies
      """

      df = pd.DataFrame(columns =  ['value', 'date', 'tag', 'zone', 'unit'])

      try:  # Check if the search term exists
         tags = eq.instances.tags(search_term)
      except: # Continue if the search term does not exist
            print("> No series found..")
            return df
      for tag in tags:
             
             for frequency in frequencies:
               try:
                  search_term_data = eq.instances.relative(
                     search_term,
                     begin = get_datetime(time_kwargs.get('start_year'), time_kwargs.get('start_month'), time_kwargs.get('start_day'), tz = CET),
                     end   = get_datetime(time_kwargs.get('end_year'), time_kwargs.get('end_month'), time_kwargs.get('end_day'), tz = CET),
                     tag = tag,
                     days_ahead = 3,
                     issued = 'latest',
                     frequency=frequency
                  ).to_dataframe()
                  search_term_data['date'] = search_term_data.index
                  search_term_data         = search_term_data.reset_index(drop=True)
                  search_term_data['tag']  = tag
                  search_term_data['zone'] = zone
                  search_term_data['unit'] = series_name
                  search_term_data.columns =  ['value', 'date', 'tag', 'zone', 'unit']
                  df = pd.concat([df,search_term_data])
                  break
               except:
                  print('> Series does not have specified frequency')
                  continue


      return df



def retrieve_all_series(zone_lists = nordics, types = ['Forecast', 'Actual', 'Backcast', 'Synthetic','REMIT'] \
                       , frequencies = ['PT1H', 'P1D'], verbose = True, **time_kwargs):
      """

      For all zones, types and frequencies, collects series into a single dataframe by
      calling iteratively on retrieve_instance()
      A type: Types of data, could be backcast actual and REMIT messages
      """

      if time_kwargs.get('start_year') is None:
             raise KeyError("Missing time kwargs")
      df = pd.DataFrame()
      for type in types:
            for zone in zone_lists:
                     print('> Starting new zone...')
                     try: 
                        series_list = eq.metadata.curves(area = zone, \
                                                         data_type= type, \
                                                         page_size=250)
                        series_number = 0    
                     except:
                        print("> No such area")
                        continue 
                     for series in series_list:
                           search_term = str(series)
                           #series_name = '_'.join(search_term.split()[0:2])
                           #zone = unit = ''.join(search_term.split()[0])
                           unit = ''.join(search_term.split()[3])
                           if type == 'Forecast':
                              series = retrieve_timeseries(search_term, unit, zone, frequencies, **time_kwargs)
                           elif type == 'REMIT':
                              series = retrieve_remit(search_term, unit, zone, **time_kwargs)
                           else:
                              series = retrieve_instance(search_term, unit, zone, frequencies, **time_kwargs)
                              
                           series['search_term'] = search_term
                           series_number = series_number + 1
                           if  verbose:
                              print(" > Adding new series") 
                              print(series_number, " of ", len(series_list))
                           df = pd.concat([df,series])

      return df





time_kwargs = {'start_year':2018, 'start_month':3, 'start_day':16, 'end_year':2022, 'end_month':4, 'end_day':19}
all_synthetic = retrieve_all_series(types = ['Synthetic'], **time_kwargs)

timeseries_test = retrieve_timeseries('NO1 Consumption Index Chilling % 15min Forecast', '%', 'NO',['PT1H', 'P1D'], **time_kwargs )

def save_data(folder_path, **time_kwargs):
      """
      Save all eq_data
      """

      all_forecasts = retrieve_all_series(types = ['Forecast'], **time_kwargs)
      all_forecasts.to_pickle(folder_path + "/forecasts.pkl")

      all_backcasts = retrieve_all_series(types = ['Backcast'], **time_kwargs)
      all_backcasts.to_pickle(folder_path+ "/backcasts.pkl")


      all_remit = retrieve_all_series(types = ['REMIT'], **time_kwargs)
      all_remit.to_pickle(folder_path + "/remits.pkl")

      all_synthetic = retrieve_all_series(types = ['Synthetic'], **time_kwargs)
      all_synthetic.to_pickle(folder_path + "/synthetic.pkl")

      all_actual = retrieve_all_series(types = ['Actual'], **time_kwargs)
      all_actual.to_pickle(folder_path+ "/actual.pkl")

       
save_data(folder_path="../data/eq_data", **time_kwargs)




def load_data():

   all_forecasts = (pd.read_pickle("../data/eq_data/forecasts.pkl"))
   all_backcasts= (pd.read_pickle("../data/eq_data/forecasts.pkl"))
   all_synthetic = (pd.read_pickle("../data/eq_data/forecasts.pkl"))



all_forecasts['date'] = pd.to_datetime(all_forecasts.date, utc = True)

no2_temp = all_forecasts.copy()[all_forecasts.copy()['search_term'] == 'NO2 Consumption Temperature °C 15min Forecast']
no2_temp['date'] = pd.to_datetime(no2_temp.date, utc = True).dt.tz_convert('Europe/Berlin')



######################### For manual exchange data retrieval
### List all possible combinations of exchanges

def exchange_data():
      """
      For manual exchange data retrieval
      """
      exchange_area_permutation = pd.DataFrame(list(itertools.permutations(all_zones, 2)))

      search_term_zones =  exchange_area_permutation[0] + '>' + exchange_area_permutation[1]
      search_term_zones = search_term_zones.to_list()

      exchange_data_forecast = retrieve_timeseries_for_all_zones(' Exchange Day-Ahead Schedule MWh/h H Forecast', 'mwh_exchanged', search_term_zones,  'PT1H')
      exchange_data_forecast.to_pickle("../data/exchange_forecast.pkl")
      return exchange_data_forecast










