"""

Script for retrieving data for use by forecasting models

"""
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from datetime import timedelta
##### Importing supporting scripts #######
# Import analysis.py 
from analysis import iterator_hour
# Modules
import pandas as pd
from termcolor import colored 
from utilities import generate_time_lags, scale_features, inverse_scale_features, rolling_mean, get_first_non_na_row
import numpy as np
import os
from datetime  import date, datetime
import gc
from utilities import * 
import logging
"""

Explanation for use:

Raw data in eq_data folder is loaded 
by filter_series_and_tags. 
This wrapper function calls on treat_df_only_priority_tags_and_series
to leave out any unwanted series and includes only a single tag of a given series, should there be multiple.
This is done according to a priority tag list.
The end result is a significantly reduced set of pickles, which can be loaded and using the Read_data class.

"""



def series_tag_priority(series : pd.DataFrame, priority_tag_list:list) -> pd.DataFrame:
        """
        Returns a single series of a dataframe composed of multiple series (tags).
        Takes the first of of an input priority and iterates through the list.
        If only a single tag exists in the list, return a dataframe composed of only that tag
        """
        if  len(series['tag'].unique()) == 1:
            print("> Only one tag in series")
            return series
        for forecast_type in priority_tag_list:
            filtered_series = series[series['tag'] == forecast_type]
            if len(filtered_series) != 0:
                return filtered_series
        if len(filtered_series) == 0:
            return None

def treat_df_only_priority_tags_and_series(df: pd.DataFrame, series_list: list,  priority_tag_list: list):
    """

    """
    filtered_df = pd.DataFrame()
    for series in series_list:
        single_series_tag = series_tag_priority(df[df.search_term == series], priority_tag_list)
        if single_series_tag is None:
            continue
        filtered_df = pd.concat([single_series_tag, filtered_df])
        del(single_series_tag)
    return filtered_df



def filter_series_and_tags(series_list: list, priority_tag_list: list) -> None:

    """
    Filters and 
    Has to be run each time news series are added
    """

    # Check if folder exists

    if not os.path.exists("../data/filtered_eq_data/"):
          os.makedirs("../data/filtered_eq_data/")
          print(colored("> Created new directory", "green"))

      
    all_forecasts   = pd.read_pickle("../data/eq_data/forecasts.pkl")
    all_forecasts = treat_df_only_priority_tags_and_series(all_forecasts, series_list, priority_tag_list=priority_tag_list)
    all_forecasts.to_pickle("../data/filtered_eq_data/filtered_forecasts.pkl")
    print(colored("> Finished retrieving forecasts", 'blue'))

    all_actual  = pd.read_pickle("../data/eq_data/actual.pkl")
    all_actual = treat_df_only_priority_tags_and_series(all_actual, series_list, priority_tag_list=priority_tag_list)
    all_actual.to_pickle("../data/filtered_eq_data/filtered_actual.pkl")
    print(colored("> Finished retrieving actuals", 'blue'))
    del(all_forecasts, all_actual)


    all_remit   = pd.read_pickle("../data/eq_data/remits.pkl")
    all_remit = treat_df_only_priority_tags_and_series(all_remit, series_list, priority_tag_list=priority_tag_list)
    all_remit.to_pickle("../data/filtered_eq_data/filtered_remits.pkl")
    print(colored("> Finished retrieving remits", 'blue'))
    

    all_synthetic   = pd.read_pickle("../data/eq_data/synthetic.pkl")
    all_synthetic = treat_df_only_priority_tags_and_series(all_synthetic, series_list, priority_tag_list=priority_tag_list)
    all_synthetic.to_pickle("../data/filtered_eq_data/filtered_synthetic.pkl")
    print(colored("> Finished retrieving synthetic", 'blue'))


# Priority series tags
priority_tag_list =    ['prefix',
                        'ecsr-ens',
                        'ec-ens',
                        'gfs-ens',
                        'ecsr',
                        'ec',
                        'arome',
                        'ukmo',
                        'ec-ext',
                        'gfs-ext']

class Read_data():

    """
    Class handling reading, processing and feature engineering needed
    for model building
    """
    def __init__(self, zone, start_date, end_date, test_mode = False):
        """
        Constructor method
        """
        self.zone       = zone
        self.test_mode  = test_mode
        self.start_date = start_date
        self.end_date   = end_date

        self.zone_price_volume_data = self.__read_price_and_volume()
        self.energy_quantified_data = self.__energy_quantified_data()
        self.uk_capacity_data = self.__uk_capacity_data(start_date, end_date)

        # Define logging tools
        logging.basicConfig(filename="../model_data/model_data_log.txt",
                        level=logging.DEBUG,
                        format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

        
    def __read_price_and_volume(self):
        
        """
        Reads raw data.
        Combines temperature and price data
        """
        elspot_zone_data              = self.__filter_zone(pd.read_pickle("../output_data/elspot_price_and_volume_2019_to_2022.pkl"), 'area', self.zone)
        elbas_zone_data               = self.__filter_zone(pd.read_pickle("../output_data/elbas_price_and_volume_2019_to_2022.pkl"), 'area', self.zone)
        if self.zone != "":
            elspot_zone_data         = elspot_zone_data[elspot_zone_data['area'] == self.zone]
            elbas_zone_data          = elbas_zone_data[elbas_zone_data['area'] == self.zone]
        elbas_zone_data = elbas_zone_data.rename(columns = {'price':'intraday_price',  'volume':'intraday_volume'})
        elspot_zone_data = elspot_zone_data.rename(columns = {'price':'dayahead_price', 'volume':'dayahead_volume'})
        zone_data = elspot_zone_data.merge(elbas_zone_data, how = 'left')
        zone_data['datetime'] = self.__convert_cet_datetime(zone_data.datetime)
        zone_data = zone_data[(zone_data['datetime'] >= self.start_date) & (zone_data['datetime'] <= self.end_date)]
        return zone_data


    def __uk_capacity_data(self, start_date, end_date):
        """
        Retrieving UK capacity data
        """
        uk_exchange_capacity   = pd.read_pickle("../output_data/uk_coupling_capacity_2019_to_2022.pkl")
        uk_exchange_capacity = uk_exchange_capacity.pivot_table(index = ['datetime'], columns = ['from_area'], values = 'exchange_capacity').reset_index()
        uk_exchange_capacity.columns.name = 'index'
        uk_exchange_capacity['net_exchange'] = uk_exchange_capacity.GB - uk_exchange_capacity.NO2

        uk_exchange_capacity = uk_exchange_capacity.rename(columns = {'net_exchange':'GB>NO2 Exchange Day-Ahead Capacity MW H Actual'})

        all_dates = iterator_hour(pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S'), pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S'))
        all_dates = pd.DataFrame({'datetime':[date for date in all_dates]})
    
        uk_exchange_capacity = all_dates.merge(uk_exchange_capacity, how = 'left')
        uk_exchange_capacity = uk_exchange_capacity.fillna(0)
        uk_exchange_capacity = uk_exchange_capacity.drop(columns = ['GB', 'NO2'])
        #print("> Before date change", uk_exchange_capacity.datetime)
        uk_exchange_capacity.datetime = self.__convert_cet_datetime(uk_exchange_capacity.datetime)
        #print("> After date change", uk_exchange_capacity.datetime)
        return uk_exchange_capacity  


    def __energy_quantified_data(self):
        """
        Retrieving eq data and storing as class variable for further use
        """
        print(colored("> Starting retrieving eq data", 'green'))
        if self.test_mode:
            path = "../model_data/test_sample/"
            if not os.path.exists(path):
                os.makedirs(path)
                print(colored("> Created new directory " + path, "green"))
            energy_quantified_data = pd.read_pickle(path + "test_eq_data.pkl")
        else:
            # Retrieve filtered series
            all_forecasts   = pd.read_pickle("../data/filtered_eq_data/filtered_forecasts.pkl")
            all_actual      = pd.read_pickle("../data/filtered_eq_data/filtered_actual.pkl")
            all_synthetic   = pd.read_pickle("../data/filtered_eq_data/filtered_synthetic.pkl")
            all_remit       = pd.read_pickle("../data/filtered_eq_data/filtered_remits.pkl")
            energy_quantified_data = pd.concat([all_forecasts, all_synthetic, all_actual,  all_remit])
            del(all_forecasts, all_synthetic, all_actual,  all_remit) # Delete from memory
        # Create a unique series ID based on search term and tag
        energy_quantified_data['series_id'] = energy_quantified_data['search_term'] + " : " + energy_quantified_data['tag']
        gc.collect() # Perform garbage collection

        energy_quantified_data = energy_quantified_data.rename(columns = {'date':'datetime'})
        energy_quantified_data['datetime'] = pd.to_datetime(energy_quantified_data['datetime'],  format='%Y-%m-%d %H:%M:%S')
       
        #energy_quantified_data['datetime'] = self.__convert_cet_datetime(energy_quantified_data.datetime)
        
        #pd.to_datetime(energy_quantified_data['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Europe/Berlin')
        energy_quantified_data = energy_quantified_data[(energy_quantified_data['datetime'] >= self.start_date) & (energy_quantified_data['datetime'] <= self.end_date)]
        
        return energy_quantified_data

    def __convert_cet_datetime(self, datetime : pd.Series):
        """
        Converts a pandas series in naive timezone format to cet pytz format (datetime64[ns, Europe/Berlin])
        Assumes that the input series is CET formatted.
        """
        datetime = pd.to_datetime(datetime,  format='%Y-%m-%d %H:%M:%S')
        utc_dates = datetime.dt.tz_localize('UTC')

        cet_dates = utc_dates.dt.tz_convert('Europe/Berlin') + timedelta(hours = -1)
        return cet_dates

    def assemble_data(self, series_list:list, priority_tag_list): 
        """
        Filters relevant search term and forecast tags and merges them
        with the nordpool price and volume data.
        Calls on series_tag_priority iteratively to create a list of unique series id (search_term + tag).
        """
        print(colored("> Starting data assembly", 'green'))
        included_series_id = []
        for series in series_list:
            #print(">New series")
            included_series_id.append(self.__series_tag_priority(
                            self.energy_quantified_data[self.energy_quantified_data['search_term'] == series], # Get column name from search term, and store value
                            series,  
                            priority_tag_list))

        # Filtered series to wide
        filtered_series = self.energy_quantified_data[self.energy_quantified_data['series_id'].isin(included_series_id)]
        filtered_series_wide = filtered_series.pivot_table(index = ['datetime'], columns = 'series_id', values = 'value').reset_index()
        filtered_series_wide.columns.name = 'index'
        
        all_relevant_series_merged  = (filtered_series_wide.merge(self.zone_price_volume_data, how='left')).drop('area', 1)
        all_relevant_series_merged.datetime = pd.to_datetime(all_relevant_series_merged.datetime, format='%Y-%m-%d %H:%M:%S')
        
        all_relevant_series_merged  = all_relevant_series_merged.merge(self.uk_capacity_data, how='left')
        
        # remove dayahead spot price
        all_relevant_series_merged = all_relevant_series_merged.drop(columns = ['NO2 Price Spot EUR/MWh H Forecast : prefix', 'NO2 Price Spot Short-term EUR/MWh H Forecast : prefix'])
        self.final_series = all_relevant_series_merged

        gc.collect() # Perform garbage collection
        self.__feature_engineering()


    def retrieve_and_save_data(self, savename):

        """
        Retrieves elspot and elbas data along with energy quantified data
        """

        if self.final_series is None:
            raise ValueError("No series")
        self.final_series['intraday_price_difference'] =  self.final_series['dayahead_price'] - self.final_series['intraday_price'] 
        self.final_series.to_pickle("../model_data/" + savename  + ".pkl")
        self.final_series.to_csv("../model_data/" + savename  + ".csv")
        return self.final_series


    def __series_tag_priority(self, series : pd.DataFrame, search_term, priority_tag_list:list) -> str:
        """
        Returns a single series of a dataframe composed of multiple series (tags).
        Takes the first of of an input priority and iterates through the list.
        If only a single tag exists in the list, return a dataframe composed of only that tag
        """
        #print("series tag", series['tag'])
        if  len(series['tag'].unique()) == 1:
            #print("> Only one tag in series")
            return series['series_id'].unique()[0]
        for forecast_type in priority_tag_list:
            #print("> Forecast type", forecast_type)
            filtered_series = series[series['tag'] == forecast_type]
            if len(filtered_series) != 0:
                return filtered_series['series_id'].unique()[0]
        if len(filtered_series) == 0:
            print("> No series with included tags found")
            logging.info("Missing series {}".format(search_term))


    def treat_NA(self, func, column_names, **kwargs):
        """
        Treats NA for all column names in final_series according to an input function 
        """
        for column in column_names:
            self.final_series[column] = func(self.final_series[column], **kwargs)


    def __filter_zone(self, df, zone_column: str, zone: str) -> pd.DataFrame:
        """
        Helper method for filtering zones
        """
        return df[df[zone_column] == zone]

    def save_eq_forecasts(self):
        """
        Save energy quantified dayahead forecasts (spot price)
        """
        no2_dayahead_eq_forecast = self.energy_quantified_data[self.energy_quantified_data['search_term'] == 'NO2 Price Spot EUR/MWh H Forecast']
        series_id = no2_dayahead_eq_forecast.series_id.unique()[0]
        no2_dayahead_eq_forecast[series_id] = no2_dayahead_eq_forecast.value
        no2_dayahead_eq_forecast = no2_dayahead_eq_forecast[['datetime', series_id]]

        no2_dayahead_eq_forecast.datetime = pd.to_datetime(no2_dayahead_eq_forecast.datetime, format='%Y-%m-%d %H:%M:%S')
        no2_dayahead_eq_forecast = no2_dayahead_eq_forecast.reset_index(drop = True)
        no2_dayahead_eq_forecast.to_csv("../model_data/eq_dayahead_forecasts.csv")


        no2_short_term_dayahead_eq_forecast = self.energy_quantified_data[self.energy_quantified_data['search_term'] == 'NO2 Price Spot Short-term EUR/MWh H Forecast']
        series_id = no2_short_term_dayahead_eq_forecast.series_id.unique()[0]
        no2_short_term_dayahead_eq_forecast[series_id] = no2_short_term_dayahead_eq_forecast.value
        no2_short_term_dayahead_eq_forecast = no2_short_term_dayahead_eq_forecast[['datetime', series_id]]

        no2_short_term_dayahead_eq_forecast.datetime = pd.to_datetime(no2_short_term_dayahead_eq_forecast.datetime, format='%Y-%m-%d %H:%M:%S')
        no2_short_term_dayahead_eq_forecast = no2_short_term_dayahead_eq_forecast.reset_index(drop = True)
        no2_short_term_dayahead_eq_forecast.to_csv("../model_data/eq_short_term_dayahead_forecasts.csv")
    
    def __feature_engineering(self):
        """
        Private method
        Method for creating new features from existing elbas and elspot price data
        """
        self.final_series.datetime        = pd.to_datetime(self.final_series.datetime)
        self.final_series                 = generate_time_lags(self.final_series, 'intraday_price', 100)
        self.final_series                 = generate_time_lags(self.final_series, 'dayahead_price', 100)
        self.final_series['seq']          = np.arange(0, len(self.final_series)).tolist()
        self.final_series                 = self.final_series.reset_index(drop = True)
        self.final_series['hour']         = self.final_series.datetime.dt.hour.astype(str)
        self.final_series['day']          = self.final_series.datetime.dt.day_of_week.astype(str)
        self.final_series['week']         = self.final_series.datetime.dt.weekofyear.astype(str)
        self.final_series['month']        = self.final_series.datetime.dt.month_name().astype(str)
        self.final_series['covid']        = pd.Series(map(lambda x: "Yes" if  ( (x > pd.to_datetime("2020-03-12", utc=True)) & (x < pd.to_datetime("2021-09-25", utc=True))) else "-",  self.final_series.datetime))
        self.final_series['nordlink']     = pd.Series(map(lambda x: "Yes" if  x > pd.to_datetime("2021-03-31", utc=True) else "-",  self.final_series.datetime))
        self.final_series['northsealink'] = pd.Series(map(lambda x: "Yes" if  x > pd.to_datetime("2021-10-01", utc=True) else "-",  self.final_series.datetime))
        

        dk_non_wind_power_sourcelist = \
                                ['DK1 CHP Central Power Production MWh/h H Actual : non-forecast',
                                'DK1 CHP Decentral Power Production MWh/h H Actual : non-forecast', 
                                'DK1 Hard Coal Power Production MWh/h H Actual : non-forecast',
                                'DK1 Natural Gas Power Production MWh/h H Actual : non-forecast',
                                'DK1 Biomass Power Production MWh/h H Actual : non-forecast',
                                'DK1 Oil Power Production MWh/h H Actual : non-forecast',
                                'DK1 Other Power Production MWh/h H Actual : non-forecast',
                                'DK1 Waste Power Production MWh/h H Actual : non-forecast']
    
        self.final_series['dk_power_non_wind'] =  self.final_series[dk_non_wind_power_sourcelist].sum(axis = 1)
        self.final_series = self.final_series.loc[:, ~self.final_series.columns.isin(dk_non_wind_power_sourcelist)]  
        # Remove non_forecast
        return self.final_series
  

# Series to be retrieved from Energy Quantified datasets

energy_quantified_series_forecasts= [

    # spot price
    'NO2 Price Spot EUR/MWh H Forecast',
    'NO2 Price Spot Short-term EUR/MWh H Forecast',
    'NO2 Hydro Reservoir Water Filling % D Forecast',

    'DK1 Solar Photovoltaic Production MWh/h 15min Forecast',
    'DK1 Wind Power Production MWh/h 15min Forecast',

    'DK1 Residual Production Day-Ahead MWh/h H Forecast',
    'DK1 Residual Load MWh/h 15min Forecast',
    'NO2 Residual Production Day-Ahead MWh/h H Forecast',
    'NO2 Residual Load MWh/h 15min Forecast',
    'NO1 Residual Production Day-Ahead MWh/h H Forecast',
    'NO1 Residual Load MWh/h 15min Forecast',
    'NO5 Residual Production Day-Ahead MWh/h H Forecast',
    'NO5 Residual Load MWh/h 15min Forecast'
    'NO5 Residual Production Day-Ahead MWh/h H Forecast',
    'NO5 Residual Load MWh/h 15min Forecast',

    'NO2 Consumption Temperature °C 15min Forecast',
    'DK1 Consumption Temperature °C 15min Forecast',
    'NO5 Consumption Temperature °C 15min Forecast',
    'NO1 Consumption Temperature °C 15min Forecast',
    'NO2 Consumption MWh/h 15min Forecast',
    'NO1 Consumption MWh/h 15min Forecast',
    'NO5 Consumption MWh/h 15min Forecast',
    'DK1 Consumption MWh/h 15min Forecast',
    'NO2 Consumption Index Chilling % 15min Forecast',
    'NO2 Consumption Index Heating % 15min Forecast',
    'NO2 Consumption Index Cloudiness % 15min Forecast',
    'DK1 Consumption Index Cloudiness % 15min Forecast',
    
    'NO2 Hydro Precipitation Energy MWh H Forecast',
    
    'NO2 Hydro Run-of-river Production MWh/h 15min Forecast',
    'NO2 Wind Power Production MWh/h 15min Forecast',
    'DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast',
    'DK1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast',
    'NO1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast', 
    'NO5>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast',
    'NL>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast',
    'GB>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast',
    'SE3>NO1 Exchange Day-Ahead Schedule MWh/h H Forecast'
]


energy_quantified_series_actuals = [
    'NO1 Hydro Power Production MWh/h H Actual',
    'NO2 Hydro Power Production MWh/h H Actual',
    'NO5 Hydro Power Production MWh/h H Actual',
    'NO2 Price Regulation Down EUR/MWh H Actual',
    'NO2 Price Regulation Up EUR/MWh H Actual',
    'DK1 Volume Regulation Net MWh H Actual',
    'DK1 Price Imbalance Consumption EUR/MWh H Actual',
    'DK1 CHP Central Power Production MWh/h H Actual',
    'DK1 CHP Decentral Power Production MWh/h H Actual',
    'DK1 Hard Coal Power Production MWh/h H Actual',
    'DK1 Natural Gas Power Production MWh/h H Actual',
    'DK1 Biomass Power Production MWh/h H Actual',
    'DK1 Oil Power Production MWh/h H Actual',
    'DK1 Other Power Production MWh/h H Actual',


    'NO1>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'NO5>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'DK1>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'NO5>NO1 Exchange Day-Ahead Capacity MW H Actual',


    'NO2>DE Exchange Day-Ahead Capacity MW H Actual',
    'DE>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'NO2>NL Exchange Day-Ahead Capacity MW H Actual',
    'NO2>GB Exchange Implicit Capacity MW H Actual',
    'SE3>NO1 Exchange Day-Ahead Capacity MW H Actual',
    'GB>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'NL>NO2 Exchange Day-Ahead Capacity MW H Actual',
    'NO2>DK1 Exchange Day-Ahead Capacity MW H Actual',
    'NO2>NO1 Exchange Day-Ahead Capacity MW H Actual',
    'NO2>NO5 Exchange Day-Ahead Capacity MW H Actual',
    'DK1 CHP Central Power Production MWh/h H Actual',
    'DK1 CHP Decentral Power Production MWh/h H Actual',
    'DK1 Hard Coal Power Production MWh/h H Actual',
    'DK1 Natural Gas Power Production MWh/h H Actual',
    'DK1 Biomass Power Production MWh/h H Actual',
    'DK1 Oil Power Production MWh/h H Actual',
    'DK1 Other Power Production MWh/h H Actual',
    'DK1 Waste Power Production MWh/h H Actual',
    'NO1 Price Imbalance Consumption EUR/MWh H Actual',
    'NO1 CHP Power Production MWh/h H Actual',
    'NO2 Volume Regulation Net MWh H Actual',
    'NO2 CHP Power Production MWh/h H Actual',
    'NO5 Price Imbalance Consumption EUR/MWh H Actual',
    'NO5 Volume Regulation Net MWh H Actual',
    'NO5 CHP Power Production MWh/h H Actual',
    'DK1 Waste Power Production MWh/h H Actual',
    'NO1 Price Imbalance Consumption EUR/MWh H Actual',
    'NO1 CHP Power Production MWh/h H Actual',
    'NO2 Volume Regulation Net MWh H Actual',
    'NO2 CHP Power Production MWh/h H Actual',
    'NO5 Price Imbalance Consumption EUR/MWh H Actual',
    'NO5 Volume Regulation Net MWh H Actual',
    'NO5 CHP Power Production MWh/h H Actual']



energy_quantified_series_remit = [
                    'NO2 Hydro Reservoir Capacity Available MW REMIT',
                    'NO2 Hydro Run-of-river Capacity Available MW REMIT',
                    'GB>NO2 Exchange Net Transfer Capacity MW 15min REMIT',
                    'DE>NO2 Exchange Net Transfer Capacity MW 15min REMIT',
                    'DK1>NO2 Exchange Net Transfer Capacity MW 15min REMIT',
                    'NO5>NO2 Exchange Net Transfer Capacity MW 15min REMIT',
                    'NO1>NO2 Exchange Net Transfer Capacity MW 15min REMIT'

                    ]


energy_quantified_series_synthetic = ['NO2 Hydro Reservoir Production MWh/h H Synthetic',
                                      'NO5 Hydro Reservoir Production MWh/h H Synthetic',
                                      'NO1 Hydro Reservoir Production MWh/h H Synthetic',
                                      'NO1 Hydro Precipitation Energy MWh H Synthetic',
                                      'NO2 Hydro Precipitation Energy MWh H Synthetic',
                                      'NO5 Hydro Precipitation Energy MWh H Synthetic']




energy_quantified_series = energy_quantified_series_forecasts + energy_quantified_series_actuals + energy_quantified_series_remit +energy_quantified_series_synthetic 




if __name__ == "__main__":


    """ Uncomment to add more series"""


    #filter_series_and_tags(energy_quantified_series, priority_tag_list)
    
    read_data = Read_data('NO2', '2019-03-15 00:00:00', '2022-03-18 00:00:00', test_mode=False)

    read_data.assemble_data(energy_quantified_series, priority_tag_list)

    # Save No2 forecasts
    read_data.save_eq_forecasts()

    # Replacing 0 with NA for columns which it makes sense
    read_data.treat_NA(lambda x: x.fillna(0), 
            column_names = [ 
            'DK1>NO2 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'DK1>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
            'DE>NO2 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
            'NL>NO2 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NL>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
            'GB>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast : prefix',
            'NO2>DE Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO2>DK1 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO2>NL Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO2>NO1 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO1>NO2 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO2>NO5 Exchange Day-Ahead Capacity MW H Actual : non-forecast',
            'NO5>NO1 Exchange Day-Ahead Capacity MW H Actual : non-forecast'])

    # Variables which makes sense to pad NA's

    padded_variables =  [
                          'NO2 Hydro Reservoir Water Filling % D Forecast : ec-ens',
                          'NO2 Hydro Run-of-river Production MWh/h 15min Forecast : ']
    read_data.treat_NA(lambda x: x.interpolate(method = 'pad'), 
                    column_names = padded_variables)

    read_data.treat_NA(lambda x: x.fillna(get_first_non_na_row(x)), 
                    column_names = padded_variables)

    selected_forecasts = read_data.retrieve_and_save_data('selected_series')

    




