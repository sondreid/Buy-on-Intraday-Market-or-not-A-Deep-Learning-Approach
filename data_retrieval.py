#################################################################################################################################
##### Modules ###################################################################################################################
#################################################################################################################################

#import tempfile
from itertools import groupby
from math import prod
from matplotlib import ticker
import pandas as pd
import os
import numpy as np
import functools
import gc
import platform
import datetime as dt
from dateutil import rrule
from sqlalchemy import func
from termcolor import colored # Colored output in terminal

from analysis import date_difference, iterator_hour


system = platform.system().lower()                               # Record system (Mac or Window)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





#################################################################################################################################
##### General functions #########################################################################################################
#################################################################################################################################

def extract_csv_file(filepath):
    """
    Functon which reads a csv files, loads it as a Pandas dataframe and returns all columns
    except excluded columns
    """
    df = pd.read_csv(filepath)
    return df



def extract_csv_file_uk(filepath):
    # Find how many rows to skip
    with open(filepath, encoding="iso-8859-1") as f:
        i = 0
        i_list = []
        first = True
        colname = []
        for line in f:
            i += 1
            if line.count(",") > 30 and first:
                colname = line.split(",")
                first = False
            if line.count(",") > 30:
                i_list.append(i)                                 # Appending the rownumber where there are more than 30 values (semicolons)

    df = pd.read_csv(
        filepath, 
        encoding = "iso-8859-1",
        skiprows = i_list[1] - 1, 
        header = None,
        decimal = ",",
        on_bad_lines='skip') 

    # Appending an extra column if the file miss "sum" column from sdv
    if len(colname) > len(df.columns):
        df[len(df.columns)] = ""
        
    df.columns = colname
    df.columns = df.columns.str.lower()
    df = df.iloc[:,:-1]

    return df



def gather_csv_data(
    path="data/Nordpool/Elspot/UK/Market_coupling_capacity", 
    ignore_folders = np.arange(1999,2019).tolist(),
    verbose = False):
    """
    Retrieves all csv files in input paths and outputs them as a list
    """
    print(colored(">> Gathering csv data pending...", "blue"))                         # Console message
    df = pd.DataFrame()
    paths = gather_paths(path = path, ignore_folders=ignore_folders, verbose = verbose)
    for filepath in paths:
        df_temp = extract_csv_file(filepath = filepath)
        df = pd.concat([df, df_temp])
    print(colored(">> Gathering csv data complete...", "blue"))                         # Console message
    return df


def gather_csv_data_uk(
    path="data/Nordpool/Elspot/UK/Market_coupling_capacity", 
    ignore_folders = np.arange(1999,2019).tolist(),
    verbose = True):
    """
    Retrieves all csv files in input paths and outputs them as a list
    """
    print(colored(">> Gathering csv data pending...", "blue"))                         # Console message
    df = pd.DataFrame()
    paths = gather_paths(path = path, ignore_folders=ignore_folders, verbose = verbose)
    for filepath in paths:
        if verbose : print("> Filepath", filepath)
        if ".csv" in filepath:
            df_temp = extract_csv_file_uk(filepath = filepath)
            df = pd.concat([df, df_temp])
    print(colored(">> Gathering csv data complete...", "blue"))                         # Console message
    return df

def extract_sdv_file(filepath, verbose = False):
    """
    Cleaning nordpool sdv data
    Note that we take the average of 3a and 3b (3rd hour because of summer and winter time)
    """
    
    # Find how many rows to skip
    with open(filepath, encoding="iso-8859-1") as f:
        i = 0
        i_list = []
        first = True
        colname = []
        for line in f:
            i += 1
            if line.count(";") > 30 and first:
                colname = line.split(";")
                first = False
            if line.count(";") > 30:
                i_list.append(i)                                 # Appending the rownumber where there are more than 30 values (semicolons)

    df = pd.read_table(
        filepath, 
        encoding = "iso-8859-1", 
        delimiter = ";", 
        skiprows = i_list[1] - 1, 
        header = None,
        decimal = ",",
        on_bad_lines='skip')                      # iso-8859-1 encoding for Norwegian language
    
    # Console print if necessary
    if verbose:
        print("> Skip rows: " + str(i_list[1] - 1))
        print("> Length New Columnname: " + str(len(colname)))
        print("> Length Actual Columnname: " + str(len(df.columns)))

    # Appending an extra column if the file miss "sum" column from sdv
    if len(colname) > len(df.columns):
        df[len(df.columns)] = ""
        
    df.columns = colname
    df.columns = df.columns.str.lower()
    df = df.iloc[:,:-1]

    return df 


def eval_any_folder(file, folders):
    """
    If filename..
    """
    
    folders = [str(x) for x in folders]
    res = [ele for ele in folders if(ele in file)]
    return bool(res)


def gather_paths(
    path="data/Nordpool/Elspot/Elspot_file", 
    ignore_folders = np.arange(1999,2019).tolist(),
    ignore_files = ['.DS_Store', '2014_24h'],
    verbose = False):
    """
    Retrieves all sdv files in input paths and outputs them as a list
    """
    
    files = [] 
    ignore_folders = [item + "/" for item in map(str, ignore_folders)]
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            file_path = dirpath+'/'+file
            if not eval_any_folder(file_path, ignore_folders + ignore_files):
                if verbose: print(file_path)
                files.append(file_path)
    return files


def gather_data(function,
    path = "data/Nordpool/Elbas/Elbas_ticker_data",
    ignore_folders = np.arange(1999,2019).tolist(),
    verbose = False):
    """
    Reads a list of csv files and returns them as a row-merged dataframe
    """
    print(colored(">> Gathering csv data pending...", "blue"))                         # Console message
    paths = gather_paths(path = path, ignore_folders=ignore_folders, verbose = verbose)
    fixed_argument_read_csv = functools.partial(function) 
    df = pd.concat(map(fixed_argument_read_csv, paths))
    print(colored(">> Gathering csv data complete...", "blue"))                         # Console message
    return df


def gather_sdv_data(
    path="data/Nordpool/Elspot/Elspot_file", 
    ignore_folders = np.arange(1999,2019).tolist(),
    verbose = False):
    """
    Retrieves all sdv files in input paths and outputs them as a list
    """
    print(colored(">> Gathering sdv data pending...", "blue"))                         # Console message
    df = pd.DataFrame()
    paths = gather_paths(path = path, ignore_folders=ignore_folders, verbose = verbose)
    for filepath in paths:
        df_temp = extract_sdv_file(filepath = filepath)
        df = pd.concat([df, df_temp])
    print(colored(">> Gathering sdv data complete...", "blue"))                         # Console message
    return df




def restructure_data(df, id_variables = []):
    df = df.copy()
    df['hour3'] = df.loc[:,['hour3a', 'hour3b']].mean(axis = 1, skipna = True)
    df.drop(columns = ['hour3a', 'hour3b'], inplace = True)
    df = df.melt(id_vars=id_variables, var_name='hour', value_name='value')
    return df




#################################################################################################################################
##### Clean functions ###########################################################################################################
#################################################################################################################################

def clean_elspot_capacity_data(df, to_area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']):
    print(colored(">>> Data cleaning pending...", "blue"))                         # Console message
    df = df.copy()
    df.rename(inplace = True, columns = {'# data type':'datatype', 'date(dd.mm.yyyy)':'date'})
    area = df.alias.str.split("_", expand = True)                 # Splitting unit columns in to "from" and "to" 
    area.columns = ['from_area', 'to_area']                                # Renaming the column
    df = pd.concat([df, area], axis = 1).drop(columns = ['alias'])                         # Concat new unit columns into single data frame
    df = df.loc[df['to_area'].isin(to_area)]
    df.hour = df.hour.str.replace("hour", "")
    df.hour = df.hour.astype(int) - 1                            # Converting 1-24 hours to 0-23 hours
    df['time'] = df.date + " " + df.hour.map(str) + ":00:00"     # Creating a time column in with date and time
    df.date = pd.to_datetime(df.time, format = "%d.%m.%Y %H:%M:%S") # Converting time to datetime type
    df = df.rename(columns = {"date": "datetime"})               # Change column name
    df.drop(columns = ["time", 'code'], inplace = True)                    # Dropping time column
    df.year = df.year.astype(int)
    df.week = df.week.astype(int)
    df.day  = df.day.astype(int)
    df.hour = df.hour.astype(int)
    df = df.replace({'datatype':{'UE': 'exchange capacity', 'CR': 'capacity reduction'}})
    print(colored(">>> Data cleaning complete...", "blue"))                         # Console message
    return df


def clean_elbas_capacity_data(df, to_area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']):
    print(colored(">>> Data cleaning pending...", "blue"))                         # Console message
    df = df.copy()
    df.rename(inplace = True, columns = {'# data type':'datatype', 'date(dd.mm.yyyy)':'date'})
    area = df.alias.str.split("_", expand = True)                 # Splitting unit columns in to "from" and "to" 
    area.columns = ['from_area', 'to_area']                                # Renaming the column
    df = pd.concat([df, area], axis = 1).drop(columns = ['alias'])                         # Concat new unit columns into single data frame
    df = df.loc[df['to_area'].isin(to_area)]
    df.hour = df.hour.str.replace("hour", "")
    df.hour = df.hour.astype(int) - 1                            # Converting 1-24 hours to 0-23 hours
    df['time'] = df.date + " " + df.hour.map(str) + ":00:00"     # Creating a time column in with date and time
    df.date = pd.to_datetime(df.time, format = "%d.%m.%Y %H:%M:%S") # Converting time to datetime type
    df = df.rename(columns = {"date": "datetime"})               # Change column name
    df.drop(columns = ["time", 'code', 'timestamp(dd.mm.yyyy hh:mm:ss)'], inplace = True)                    # Dropping time column
    df.year = df.year.astype(int)
    df.week = df.week.astype(int)
    df.day  = df.day.astype(int)
    df.hour = df.hour.astype(int)
    df = df.replace({'datatype':{'IC': 'intraday capacity'}})
    print(colored(">>> Data cleaning complete...", "blue"))                         # Console message
    return df


def clean_elspot_price_and_volume_data(df, area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']):
    print(colored(">>> Data cleaning pending...", "blue"))                         # Console message
    df = df.copy()
    df.rename(inplace = True, columns = {'# data type':'datatype', 'date(dd.mm.yyyy)':'date'})
    df = df.loc[df['alias'].isin(area)]
    df.hour = df.hour.str.replace("hour", "")
    df.hour = df.hour.astype(int) - 1                            # Converting 1-24 hours to 0-23 hours
    df['time'] = df.date + " " + df.hour.map(str) + ":00:00"     # Creating a time column in with date and time
    df.date = pd.to_datetime(df.time,  format = "%d.%m.%Y %H:%M:%S") # Converting time to datetime type
    df = df.rename(columns = {"date": "datetime"})               # Change column name
    df.drop(columns = ["time"], inplace = True)                    # Dropping time column
    df.year = df.year.astype(int)
    df.week = df.week.astype(int)
    df.day  = df.day.astype(int)
    df.hour = df.hour.astype(int)
    df = df.replace(
        {'datatype':{'PR': 'prices', 
                     'OM': 'turnover'},
         'code':    {'SF': 'preliminary exhange rate',
                     'SO': 'official exchange rate',
                     'SK': 'buy volume',
                     'SS': 'sell volume'}
        })
    print(colored(">>> Data cleaning complete...", "blue"))                         # Console message
    df_price_volume_wider = df[(df['code'] == 'official exchange rate') | (df['code'] == 'buy volume')  ].copy()
    df_price_volume_wider = df_price_volume_wider[(df_price_volume_wider['unit'] == 'EUR') | (df_price_volume_wider['unit'] == 'MWh/h')]
    df_price_volume_wider = df_price_volume_wider.pivot_table(index =  ['datetime', 'alias', 'year', 'week', 'day'], columns = 'datatype', values = 'value').reset_index()

    df_price_volume_wider.columns.name = "index"

    df_price_volume_wider.columns = ['datetime', 'area', 'year', 'week', 'day', 'price', 'volume']
    

    return df_price_volume_wider


def clean_elbas_price_and_volume_data(df, area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']):
    print(colored(">>> Data cleaning pending..", "blue"))                         # Console message
    df = df[df['BArea'].isin(area)]
    print(colored(">>> Data cleaning complete...", "blue"))                         # Console message
    return df

def clean_uk_coupling_capacity_data(df):
    print(colored(">>> Data cleaning pending...", "blue"))
    df = df.copy()  
    df = df[df.alias.isin(['NO2_GB', 'GB_NO2'])]
    df = df.drop(['code', 'year', 'week', 'day'], axis = 1)
    df = df.rename(columns = {'# data type':'datatype','date(dd.mm.yyyy)':'date'})
    df.hour3a = pd.to_numeric(df.hour3a)
    df.hour3b = pd.to_numeric(df.hour3b)
    df = restructure_data(df, id_variables=['datatype', 'date', 'alias'])
    area = df.alias.str.split("_", expand = True)                 # Splitting unit columns in to "from" and "to" 
    area.columns = ['from_area', 'to_area']                                # Renaming the column
    df = pd.concat([df, area], axis = 1).drop(columns = ['alias'])
    df.hour = df.hour.str.replace("hour", "")
    df.hour = df.hour.astype(int) - 1                            # Converting 1-24 hours to 0-23 hours
    df['time'] = df.date + " " + df.hour.map(str) + ":00:00"     # Creating a time column in with date and time
    df.date = pd.to_datetime(df.time, format = "%d.%m.%Y %H:%M:%S") # Converting time to datetime type  
    df = df.rename(columns = {"date": "datetime", 'value':'exchange_capacity'})               # Change column name
    df.drop(columns = ["time"], inplace = True)                    # Dropping time column
    df = df.replace({'datatype':{'UE': 'exchange capacity', 'CR': 'capacity reduction'}})
    df.value = pd.to_numeric(df.exchange_capacity)
    df = df.drop(columns=['datatype', 'hour'])
    df = df.reset_index(drop = True)
    print(colored(">>> Data cleaning complete...", "blue"))  
    return df


#################################################################################################################################
##### Gather functions ##########################################################################################################
#################################################################################################################################

def gather_elspot_price_volume_data(
    area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], 
    from_year = 2019) -> pd.DataFrame:
    """
    Gathers elspot price and volume data
    """
    
    df_price_volume = gather_data(extract_sdv_file,
        path           = "data/Nordpool/Elspot/Elspot_file",
        ignore_folders = np.arange(1999, from_year).tolist())

    df_price_volume = restructure_data(
        df              = df_price_volume, 
        id_variables    = ['# data type', 'code', 'year', 'week', 'day', 'date(dd.mm.yyyy)', 'alias', 'unit'])
    
    df_price_volume = clean_elspot_price_and_volume_data(
        df              = df_price_volume, 
        area            = area)



    print(colored(">>>> Pickle pending...", "blue"))                                 # Console message
    df_price_volume.to_pickle("output_data/elspot_price_and_volume_" + str(from_year) + "_to_2022" + ".pkl") # Pickling data frame
    print(colored(">>>> Pickle complete...", "blue"))                                # Console message
    
    return df_price_volume





def gather_elbas_tickers(
    area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'],
    from_year = 2019):
    """
    Wrapper function for Elbas ticker data retrieval
    """

    df_elbas_tickers = gather_data(extract_csv_file,
        path = "data/Nordpool/Elbas/Elbas_ticker_data", 
        ignore_folders = np.arange(1999,from_year).tolist())

    df_elbas_tickers = clean_elbas_price_and_volume_data(df_elbas_tickers, area = area)
    
    return df_elbas_tickers


def gather_elspot_capacity_data(
    area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], 
    from_year = 2019):
    
    df_capacity = gather_data(extract_sdv_file,
        path            = "data/Nordpool/Elspot/Elspot_capacity",
        ignore_folders  = np.arange(1999, from_year).tolist())

    df_capacity = restructure_data(
        df              = df_capacity, 
        id_variables    = ['# data type', 'code', 'year', 'week', 'day', 'date(dd.mm.yyyy)', 'alias'])

    df_capacity = clean_elspot_capacity_data(
        df              = df_capacity, 
        to_area         = area)

    print(colored(">>>> Pickle pending...", "blue"))                                 # Console message
    df_capacity.to_pickle("output_data/elspot_capacity_" + str(from_year) + "_to_2022" + ".pkl") # Pickling data frame
    print(colored(">>>> Pickle complete...", "blue"))                                # Console message
    
    return df_capacity


def gather_elbas_capacity_data(
    area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], 
    from_year = 2019):
    
    df_capacity = gather_data(extract_sdv_file,
        path            = "data/Nordpool/Elbas/Elbas_capacity",
        ignore_folders  = np.arange(1999, from_year).tolist())

    df_capacity = restructure_data(
        df              = df_capacity, 
        id_variables    = ['# data type', 'code', 'year', 'week', 'day', 'date(dd.mm.yyyy)', 'alias', 'timestamp(dd.mm.yyyy hh:mm:ss)'])

    df_capacity = clean_elbas_capacity_data(
        df              = df_capacity, 
        to_area         = area)

    print(colored(">>>> Pickle pending...", "blue"))                                 # Console message
    df_capacity.to_pickle("output_data/elbas_capacity_" + str(from_year) + "_to_2022" + ".pkl") # Pickling data frame
    print(colored(">>>> Pickle complete...", "blue"))                                # Console message
    
    return df_capacity


def gather_elbas_price_and_volume(area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], from_year = 2019):
    """
    Wrapper function: Processing Volume weighted prices for elbas data, and stores it as a pickle
    """
    elbas_ticker_data = gather_elbas_tickers(area = area, from_year = from_year)
    format_date_from_utc_ticker_data = format_date_from_utc_to_cet(elbas_ticker_data,  'Trade Time UTC')
    
    print(colored(">>>> Pickle pending... elbas tickers", "blue"))                                 # Console message
    format_date_from_utc_ticker_data.to_pickle('output_data/elbas_tickers_'  + str(from_year) + '_to_2022' + '.pkl')
    
    print(colored(">>>> Pickle complete... elbas tickers", "blue")) 

    
    per_hour_and_block_bids_split = product_codes_to_dates(format_date_from_utc_ticker_data, 'Product Code')
    del(format_date_from_utc_ticker_data)   
    gc.collect() # Perform garbage collection
    elbas_volume_and_price = calculate_vwp(per_hour_and_block_bids_split)

    print(colored(">>> Pickle pending... elbas volume and price", "blue"))  
    elbas_volume_and_price.to_pickle('output_data/elbas_price_and_volume_' + str(from_year) + '_to_2022.pkl')
    print(colored(">>> Pickle complete... elbas volume and price", "blue"))  
    return elbas_volume_and_price



def gathering_uk_coupling_capacity(from_year = 2019):
    df = gather_csv_data_uk(path = "data/Nordpool/Elspot/UK/Market_coupling_capacity")
    df = clean_uk_coupling_capacity_data(df)
    print(colored(">>> Pickle pending... UK coupling capacity", "blue"))  
    df.to_pickle('output_data/uk_coupling_capacity_' + str(from_year) + '_to_2022.pkl')
    print(colored(">>> Pickle complete... UK coupling capacity", "blue"))  
    return df



#################################################################################################################################
##### Special calculation functions #############################################################################################
#################################################################################################################################

def iterator_hour(start, finish):
    """
    Creates and returns an iterater of all dates per hour
    between start and finish input dates
    """
    return rrule.rrule(rrule.HOURLY, dtstart=start, until=finish)


def format_date_from_utc_to_cet(df, column):
    """
    Takes in a dataframe and a string column, returns a dataframe with an added column
    in datetime format (Y-M-D-H)
    """
    df['buy_time'] = [str(x)[0:13] for x in df[column]]
    df['buy_time'] = pd.to_datetime(df['buy_time'], format = "%Y/%m/%d:%H", utc=True)
    df = df.drop(['Trade Time UTC'], axis = 1)
    df['buy_time'] = df['buy_time'].dt.tz_convert('Europe/Berlin')
    return df


def new_type_orders(series: pd.Series) -> bool:

    """
    Checks if a series is comprised of newer type orders on Elbas,
    which contains PH
    """
    series = series.dropna()
    per_hour_bids = ['PH' in x for x in series]
    return all(per_hour_bids) # Return boolean: if all rows are a per-hour ivds


def single_hour_product_code_to_date(date, years = [2019, 2020, 2021, 2022]):
    """
    Converts Elbas date format single row to datetime
    """
    if type(date) == int or len(date) < 6:
        print("> Missing date value")
        return None
    date =  date.replace('-', '')# Remove unwanted signs
    year = int(date[0:4])
    if year not in years:
        return None
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[8:10]) -1
    return dt.datetime(year, month, day, hour)


def split_single_custom_blocks(row, product_column, verbose  = False) -> pd.DataFrame:
    """
    Splits a custom block order from a single rows into several rows
    """
    start_hour = single_hour_product_code_to_date(row[product_column][1])
    end_hour  = single_hour_product_code_to_date(row[product_column][2])
    if verbose:
        print("> End hour", end_hour)
    custom_block_hours = iterator_hour(start_hour, end_hour)
    
    num_hours = sum(1 for _ in custom_block_hours)
    output_df = pd.DataFrame([row for x in range(0, num_hours)])
    output_df['order_time'] = [date for date in custom_block_hours]
    
    return output_df


def split_custom_blocks_to_hours(df: pd.DataFrame, product_column: str) -> pd.DataFrame:
    """
    Splits a dataframe comprised of custom-block orders into per-hour-bids
    """
    split_rows = [split_single_custom_blocks(df.iloc[x], product_column) for x in range(0, len(df))]
    combined_dataframe = pd.concat(split_rows)
    return combined_dataframe


def product_codes_to_dates(df: pd.DataFrame, product_column: str) -> pd.DataFrame:
    """
    For single hour bids, converts and formats the dates as a new column.
    For custom block orders 
    """
    assert (new_type_orders(df[product_column])), 'Series contains orders other than per-hour-bids'
    df[product_column]=  df[product_column].str.split('PH') 

    per_hour_bids = df[[len(x) < 3 for x in df['Product Code']]]
    per_hour_bids['order_time'] = [single_hour_product_code_to_date(x[1]) for x in per_hour_bids['Product Code']]
    
    custom_block_bids = df[[len(x) > 2 for x in df['Product Code']]]
    custom_block_bids = split_custom_blocks_to_hours(custom_block_bids, product_column)

    concat_df = pd.concat([custom_block_bids, per_hour_bids], ignore_index=True) # Concat single-hour and custom block bids

    return concat_df


def calculate_vwp(df):
    """
    Calculates Volume Weighted Price for each area and hour in an input dataframe
    """
    print("> Calulating volume weighted price pending...") 
    df = df[df['Cancelled'] != 1] # Filter out cancelled orders
    df['price_volume'] = df['Price']*df['QTY']
    vwp = pd.DataFrame(df.groupby(['order_time', 'BArea']).sum()['price_volume']/df.groupby(['order_time', 'BArea']).sum()['QTY'])
    order_time = [ vwp.index[x][0] for x in range(len(vwp))]
    area = [ vwp.index[x][1] for x in range(len(vwp))]
    vwp['order_time'] = order_time
    vwp['area'] = area
    vwp['volume'] = df.groupby(['order_time', 'BArea']).sum()['QTY']
    vwp.columns = ['price', 'datetime', 'area', 'volume']
    vwp = vwp.reset_index(drop=True)
    print("> Calulating volume weighted price complete...") 
    return vwp


    


if __name__ == '__main__':
    print("> Starting data retrieval....")

    # Elspot data: price and volume
    print(colored("> Reading elspot price and volume data pending....", "red"))
    gather_elspot_price_volume_data(area = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], from_year = 2019)
    print(colored("> Reading elspot price and volume data complete....", "red"))


    # Elbas data: price and volume
    print(colored("> Reading elbas price and volume data pending....", "red"))
    gather_elbas_price_and_volume(from_year = 2019)
    print(colored("> Reading elbas price and volume data complete....", "red"))


    # UK coupling capacity data: price and volume
    print(colored("> Reading UK coupling capacity data pending....", "red"))
    gathering_uk_coupling_capacity(from_year = 2019)
    print(colored("> Reading UK coupling capacity data complete....", "red"))

    print("> End of data retrieval....")












