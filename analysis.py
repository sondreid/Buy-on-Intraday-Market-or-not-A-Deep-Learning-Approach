###########################################################################################################################
##### Analysis ############################################################################################################
###########################################################################################################################

# Modules

from multiprocessing.dummy import freeze_support

import os 



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.ticker as ticker
from dateutil import rrule
import matplotlib.dates as md
from models.utilities import remove_nans
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import math
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates




###########################################################################################################################
##### General settings ####################################################################################################
###########################################################################################################################

# Theme and palette for plots
sns_theme = 'white'
sns_palette = 'Set2'
font = 'Computer Modern'
sns.set(rc={'figure.figsize':(55,15)})




###########################################################################################################################
##### Functions ###########################################################################################################
###########################################################################################################################

################################
##### Seaborn plot-functions
################################
def sns_lineplot(
    df, 
    x, 
    y, 
    xlab        = '', 
    ylab        = '', 
    xlim_min    = None, 
    xlim_max    = None,
    ylim_min    = None, 
    ylim_max    = None,  
    lw          = 3, 
    xinterval   = None, 
    offset      = None, 
    hue         = None, 
    plot_title  = '',
    title_fontsize = None,  
    file_title  = None, 
    figsize     = None, 
    fontscale   = 3, 
    save        = False, 
    legend_text = False,
    legend_loc = "upper right",
    directory   = 'images/', estimator = np.median,
    despine = 0,
    month = None,
    bigax = False,
    clf = True):
    
    '''
    Wrapper-function that performs seaborn lineplot
    '''
    if figsize is not None:
        sns.set(rc={'figure.figsize':figsize})
    sns.set_context('paper', font_scale = 7)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    
    
    ax = sns.lineplot(
        data        = df, 
        x           = x, 
        y           = y, 
        hue         = hue, 
        ci          = None,
        estimator   = estimator, 
        lw          = lw, 
        palette     = sns_palette)
    
    sns.despine(offset = despine)
    plt.legend(loc=legend_loc)
    if legend_text is not False:
        plt.legend(title = legend_text, loc='upper right', shadow = True, fontsize = 'large')

    ax.set(xlabel=xlab, ylabel=ylab)
    ax.axes.set_title(plot_title, weight='bold', fontsize=title_fontsize) # + ' for area: ' + str(df.area.unique())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    if bigax == True:
        axissize = 100

        ax.xaxis.label.set_size(axissize)
        ax.yaxis.label.set_size(axissize)

    leg = ax.legend()

    for line in leg.get_lines():
        line.set_linewidth(10)

    if month is not None:
        myFmt = mdates.DateFormatter('%Y-%b')
        ax.xaxis.set_major_formatter(myFmt)

    if xinterval != None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xinterval))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    if (xlim_min != None) or (xlim_max != None): ax.set(xlim=(xlim_min, xlim_max))    
    if (ylim_min != None) or (ylim_max != None): ax.set(ylim=(ylim_min, ylim_max)) 

    if save: plt.savefig(directory + file_title + '.png', bbox_inches='tight', edgecolor='none', facecolor="white")
    plt.show()
    if clf:
        plt.clf()


def sns_pointplot(
    df, 
    x, 
    y, 
    xlab = '', 
    ylab = '', 
    offset = None, 
    hue = None, 
    plot_title = '', 
    file_title = None, 
    figsize = (30,10), 
    fontscale=3, 
    save = False, 
    lw          = 3, 
    directory = 'images/', 
    estimator = np.median):

    '''
    Function
    '''
    
    sns.set_context('paper', font_scale = 4)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    
    ax = sns.pointplot(
        x           = x, 
        y           = y, 
        hue         = hue, 
        data        = df, 
        dodge       = True, 
        estimator   = estimator, 
        palette     = sns_palette,
        lw          = lw)

    sns.despine(offset = offset)
    plt.legend(loc='upper left')
    ax.set(xlabel=xlab, ylabel=ylab)
    ax.axes.set_title(plot_title, weight='bold')

    leg = ax.legend()

    for line in leg.get_lines():
        line.set_linewidth(10)

    #axissize = 100

    #ax.xaxis.label.set_size(axissize)
    #ax.yaxis.label.set_size(axissize)

    if save: plt.savefig(directory + file_title + '.png', bbox_inches='tight', edgecolor='none', facecolor="white")
    plt.show()
    plt.clf()


def sns_boxplot(
    df, 
    x, 
    y, 
    plot_title, 
    hue, 
    xlab        = '', 
    ylab        = '', 
    order       = None, 
    file_title  = None, 
    directory   = 'images/', 
    save        = False):
    
    sns.set_context('paper', font_scale = 4)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    
    ax = sns.boxplot(
        data    = df, 
        x       = x, 
        y       = y, 
        hue     = hue, 
        order   = order, 
        palette = sns_palette)

    ax.set(xlabel=xlab, ylabel=ylab)
    sns.despine()
    plt.legend(loc='upper left')
    plt.title(plot_title, weight='bold')

    #axissize = 100

    #ax.xaxis.label.set_size(axissize)
    #ax.yaxis.label.set_size(axissize)

    if save: plt.savefig(directory + file_title + '.png', bbox_inches='tight')
    plt.show()
    plt.clf()



def sns_barplot(df, 
    x, 
    y, 
    order = None,
    xlab        = '', 
    ylab        = '', 
    xlim_min    = None, 
    xlim_max    = None, 
    lw          = 3, 
    xinterval   = None, 
    offset      = None, 
    hue         = None, 
    plot_title  = '', 
    file_title  = None, 
    figsize     = (30,10), 
    fontscale   = 3, 
    save        = False, 
    directory   = 'images/', 
    estimator = np.median):

    sns.set_context('paper', font_scale = 7)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    
    ax = sns.barplot(
        data        = df, 
        x           = x, 
        y           = y, 
        hue         = hue, 
        ci          = None,
        estimator   = estimator, 
        lw          = lw, 
        palette     = sns_palette)
    
    sns.despine(offset = 50)
    ax.set(xlabel=xlab, ylabel=ylab)
    ax.axes.set_title(plot_title, weight='bold')

    if xinterval != None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xinterval))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    if (xlim_min != None) or (xlim_max != None): ax.set(xlim=(xlim_min, xlim_max))    

    if save: plt.savefig(directory + file_title + '.png', bbox_inches='tight', edgecolor='none', facecolor="white")
    plt.show()
    plt.clf()

def sns_histplot(
    df, 
    x, 
    plot_title, 
    hue, 
    binwidth, 
    xinterval = 1,
    xlab        = '', 
    ylab        = '', 
    file_title = None, 
    directory = 'images/', 
    save = False):
    '''
    Function
    '''
    sns.set_context('paper', font_scale = 4)
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    ax = sns.histplot(data = df, x = x, hue = hue, binwidth=binwidth)
    ax.set(xlabel=xlab, ylabel=ylab)
    sns.despine()
    if xinterval != None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xinterval))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title(plot_title, weight='bold')
    if save: plt.savefig(directory + file_title + '.png', bbox_inches='tight')
    plt.show()
    plt.clf()





################################
##### Other functions
################################

def bar_plot(df, x, y):
    '''
    '''
    fig, ax = plt.subplots(figsize = (15, 6)) 
    sns.barplot(data = df, x = x, y = y)
    fig.suptitle('Price per weekday', fontsize = 15)
    fig.tight_layout()  


def line_plot(df, x, y):
    '''
    '''
    fig, ax = plt.subplots(figsize = (15, 6)) 
    sns.set_style('white')
    sns.color_palette('tab10')
    sns.lineplot(data = df, x = x, y = y, palette= sns_palette)
    #sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
    fig.tight_layout()  


def summary_table(df, groupby):
    tabell = pd.DataFrame(df.groupby(groupby).describe().reset_index())
    return(tabell)


def density_plots(df):
    '''
    Densitiy plots of temperature and price
    '''
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(10,6))
    ax1.set_title('Density plots of temperature')
    ax1.set_title('Density plots of price')
    sns.kdeplot(df['temperature'], ax = ax1)
    sns.kdeplot(df['price'], ax = ax2)
    plt.show()

def generate_time_lag(df, value, n_lag):
    '''
    Single
    Function which lags the value of the input dataframe with n number of lags (n_lags)
    '''
    df_n = df.copy()
    df_n[f'{value}_lag{n_lag}'] = df_n[value].shift(n_lag)
    df_n = df_n.iloc[n_lag:]
    return df_n


def generate_time_lags(df, value, n_lags):
    '''
    All lags
    Function which lags the value of the input dataframe with n number of lags (n_lags)
    '''
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f'{value}_lag{n}'] = df_n[value].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def date_difference(start_date, end_date):
    '''
    Finds the difference in hours between two dates
    '''
    difference = end_date -start_date
    secs= difference.total_seconds()
    hrs,secs=divmod(secs,secs_per_hr:=60*60)
    return hrs


def iterator_hour(start, finish):
    '''
    Creates and returns an iterater of all dates per hour
    between start and finish input dates
    '''
    return rrule.rrule(rrule.HOURLY, dtstart=start, until=finish)


def date_difference_df(df, date_column):
    '''
    Finds the difference in hours between two dates
    '''

    first_date = df[date_column].min()
    end_date = df[date_column].max()
    difference = end_date -first_date
    secs= difference.total_seconds()
    hrs,secs=divmod(secs,secs_per_hr:=60*60)
    return hrs


def check_intraday_hours(df, date_column):
    ''' 
    Returns the difference between the actual hours betweenn start and end date, and the
    actual number of hours traded on Intraday
    '''

    first_date = df[date_column].min()
    end_date = df[date_column].max()
    observed_hours = len(df[date_column].unique())
    actual_hours = date_difference(first_date, end_date)
    return actual_hours - observed_hours


def cross_correlations(df, x, y, lags):
    ''' 
    Calculates cross correlation between to variables x and y for all upto all 
    lags specified by lags. 
    if x and y refers to the same series, the resulting series will be a autocorrelational series
    '''
    correlation_df = pd.DataFrame({'lags':range(0, lags)})

    correlations = [do_cross_corr(df, x, y, lag) for lag in range(0, lags)]
    correlation_df['correlation'] = correlations
    return correlation_df 


def do_cross_corr(df, x, y,  lags):
    ''' 
    Wrapper around pearsons correlation coefficient (by Pandas)
    '''
    df = df.dropna()
    return df[x].corr(df[y].shift(lags))


def non_traded_hours_aggregated(
    df, 
    areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], 
    date_column = 'datetime'):
    
    '''
    Calculatesd the missing traded hours
    '''
    
    df = df.dropna()
    non_traded_hours = [check_intraday_hours(df[df['area'] == area], date_column) for area in areas]
    actual_hours =     [date_difference_df(df[df['area'] == area], date_column) for area in areas]
    traded_hours_df = pd.DataFrame({'areas': areas, 'non_traded_hours':non_traded_hours, 'non_traded_hours_percent':np.array(non_traded_hours)/np.array(actual_hours)})
    return traded_hours_df


def long_to_wide_df(long_df: pd.DataFrame, columns = 'market', values = 'price') -> pd.DataFrame:
    '''
    
    '''
    indices = [ele for ele in list(long_df.columns) if ele not in [columns, values]]
    wide_df = long_df.pivot_table(index = indices, columns = columns, values = values).reset_index()
    wide_df.columns.name = 'index'
    return wide_df


def wide_to_long_df(wide_df : pd.DataFrame,   var_name = 'market', value_name = 'price', value_vars = ['dayahead_lag1', 'Intraday', 'Day-ahead']) -> pd.DataFrame:
    ''' 
    '''
    indices = [ele for ele in list(wide_df.columns) if ele not in value_vars]
    #value_vars = wide_df[columns].unique()
    long_df = wide_df.melt(id_vars = indices, 
                var_name = var_name, 
                value_name = value_name, value_vars = value_vars)
    return long_df


def missing_hours_per_area(traded_hours_df, area, groupby = ['year', 'weekofyear']):
    '''
    Calculates missing trading hours for a given area
    '''
    traded_hours_area = traded_hours_df[traded_hours_df['area'] == area]
    missing_hours_it = iterator_hour(traded_hours_area.datetime.min(), traded_hours_area.datetime.max())
    actual_hours = pd.DataFrame({'datetime': [date for date in missing_hours_it]})
    traded_hours_area  = actual_hours.merge(traded_hours_area, on = 'datetime', how = 'left')  
    traded_hours_area = traded_hours_area.fillna(value = {'volume': 0})
    missing_tradehours_area =   traded_hours_area[(traded_hours_area['volume'] == 0)]
    missing_tradehours_area = date_features(missing_tradehours_area)
    missing_tradehours_area.weekofyear = missing_tradehours_area.weekofyear.astype(int)
    missing_tradehours_area.year = missing_tradehours_area.year.astype(int)
    missing_tradehours_area = missing_tradehours_area.groupby(by = groupby, as_index=False).count().reset_index()
    missing_tradehours_area['index'] = missing_tradehours_area.index
    missing_tradehours_area['area'] = area
    return missing_tradehours_area


def date_features(df, date_column = 'datetime'):
    '''
    Adds new date columns derived from date column
    '''
        
    df['hour']       = df[date_column].dt.hour.astype(int)
    df['day_name']   = df[date_column].dt.day_name().astype(str)
    df['weekday']    = df[date_column].dt.weekday.astype(int)
    df['weekofyear'] = df[date_column].dt.weekofyear.astype(int)
    df['month_name'] = df[date_column].dt.month_name().astype(str)
    df['year']       = df[date_column].dt.year.astype(int)
    return df


def fill_nordpool_data(df, areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'], types = ['Intraday']):
    df = df.copy()

    #Trading volume Intraday:  Fill inn missing hours for Intraday
    missing_hours_it = iterator_hour(df.datetime.min(), df.datetime.max())
    actual_hours = pd.DataFrame({'datetime': [date for date in missing_hours_it]})

    new_df = pd.DataFrame()
    for typee in types:
        for area in areas:
            temp_df = df[df.market == typee]
            temp_df =  df[df.area == area]
            temp_df = actual_hours.merge(temp_df, on='datetime', how='left')
            temp_df.area = area
            temp_df.market = typee
            new_df = pd.concat([new_df, temp_df])
    return new_df



def count_missing_values_between_recordings(df, col, area):
    na_list = []
    count = 0
    df = pd.DataFrame(df.loc[df.area == area,col]).isnull()
    for i in range(1,len(df)-1):
        if df.iloc[i-1,0] == False and df.iloc[i+1,0] == True:
            count += 1
        if df.iloc[i-1,0] == True and df.iloc[i+1,0] == True:
            count += 1
        if df.iloc[i-1,0] == True and df.iloc[i+1,0] == False:
            na_list.append(count)
            count = 0
    na_df = pd.DataFrame(na_list).astype(str).reset_index()
    na_df.columns = ['index', 'hours']
    return na_df




def print_latex(df, table_title, float_format = "%.1f"):
    """
    
    """
    latex_code = df.to_latex(index = False,
    float_format = float_format,
    caption      = table_title,
    position     = '!h',
    label        = table_title)

    latex_list = latex_code.split("\n")
    latex_list.insert(3, "\\textbf{" + table_title + "}")
    latex_code = "\n".join(latex_list)
    print(latex_code)



def check_stationarity(series, testtype:str = 'adfuller'):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    if testtype == 'adfuller':
        # p-value < 5% -> stationary H0: non-stationary

        result = adfuller(series.values, autolag='AIC')

        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

        if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")

    if testtype == 'kpss':
        # p-value < %5 -> non-stationary H0: stationary
        result = kpss(series.values, regression='c')

        print('KPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[3].items():
            print('\t%s: %.3f' % (key, value))

        if (result[1] <= 0.05) & (result[3]['5%'] > result[0]):
            print("\x1b[31mNon-stationary\x1b[0m")
        else:
            print("\u001b[32mStationary\u001b[0m")


def __convert_cet_datetime(datetime:pd.Series) -> pd.Series:
    """
    Converts a pandas series in naive timezone format to cet pytz format (datetime64[ns, Europe/Berlin])
    Assumes that the input series is CET formatted.
    """
    datetime = pd.to_datetime(datetime, format='%Y-%m-%d %H:%M:%S')
    utc_dates = datetime.dt.tz_localize('UTC')
    cet_dates = utc_dates.dt.tz_convert('Europe/Berlin') + dt.timedelta(hours = -1)
    return cet_dates

def retrieve_predictions_cv_fold(directory:str) -> pd.DataFrame:
    df = pd.DataFrame()
    for filename in os.scandir(directory):
        if filename.is_file() and filename.name != ".DS_Store":
            print(filename.name)
            df_temp = pd.read_csv(filename.path)
            df_temp['filename'] = filename.name 
            df = pd.concat([df, df_temp])
    df.rename(columns = {'Unnamed: 0':'horizon_index'}, inplace = True)
    df.datetime = __convert_cet_datetime(df.datetime.str.split('+', expand=True).iloc[:,0])
    df['hour'] = df.datetime.dt.hour
    df.filename = df.filename.str.split('.', expand=True).iloc[:,0]
    df.horizon_index += 1 
    return df


def retrieve_predictions_cv_all_folds(directory:str) -> pd.DataFrame:
    df = pd.DataFrame()
    for filename in os.scandir(directory):
        if filename.is_file() and filename.name != ".DS_Store":
            print(filename.name)
            df_temp = pd.read_csv(filename.path)
            df_temp['filename'] = filename.name
            df_temp['fold'] = int(filename.name[-5:-4]) 
            df = pd.concat([df, df_temp])
    df = df.replace('_fold[0-9]', '', regex=True)
    df.rename(columns = {'Unnamed: 0':'horizon_index'}, inplace = True)
    df.datetime = __convert_cet_datetime(df.datetime.str.split('+', expand=True).iloc[:,0])
    df['hour'] = df.datetime.dt.hour
    df.filename = df.filename.str.split('.', expand=True).iloc[:,0]
    df.horizon_index += 1 
    df = df.replace('ETS evaluation dayahead', 'ETS(A,Ad,A)168', regex=True)
    df = df.replace('LSTM dayahead evaluation', 'LSTM', regex=True)
    df = df.replace('EQ short-term dayahead evaluation', 'EQ short-term dayahead', regex=True)
    df = df.replace('Mean dayahead evaluation', 'Mean', regex=True)
    df = df.replace('Take last dayahead evaluation', 'Naïve', regex=True)
    df = df.replace('ARIMA evaluation dayahead', 'ARIMA(5,1,5)', regex=True)
    df = df.replace('ARIMA evaluation intraday', 'ARIMA(2,1,4)', regex=True)
    df = df.replace('GRU intraday evaluation', 'GRU', regex=True)
    df = df.replace('EQ short-term intraday evaluation', 'EQ short-term dayahead', regex=True)
    df = df.replace('Mean intraday evaluation', 'Mean', regex=True)
    df = df.replace('Take last intraday evaluation', 'Naïve', regex=True)
    df = df.replace('ETS evaluation intraday', 'ETS(A,N,N)', regex=True)
    return df



def mean_mae_per_hour(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['absolute_error'] = abs(df.actuals - df.forecasts) 
    df = df.loc[:,['filename','hour','absolute_error']].groupby(['filename','hour']).mean('absolute_error').reset_index().rename(columns = {'absolute_error': 'HMAE'})
    return df

def mean_mae_per_horizon_index(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['absolute_error'] = abs(df.actuals - df.forecasts) 
    df = df.loc[:,['filename','horizon_index','absolute_error']].groupby(['filename','horizon_index']).mean('absolute_error').reset_index().rename(columns = {'absolute_error': 'HMAE'})
    return df


def mean_mae_per_fold(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['absolute_error'] = abs(df.actuals - df.forecasts) 
    df = df.loc[:,['filename','fold','absolute_error']].groupby(['filename','fold']).mean('absolute_error').reset_index().rename(columns = {'absolute_error': 'FMAE'})
    return df




def gather_production_df_and_calculate_estimated_intraday_prices(path:str="models/difference_probabilities/MLP model_prod_probabilities.csv"):
    df = pd.read_csv(path)
    df['estimate_intraday_price'] = np.nan
    df['estimate_tag'] = np.nan
    for i in range(len(df)):
        if ~np.isnan(df['intraday_actuals'][i]):
            df['estimate_intraday_price'][i] = df['intraday_actuals'][i]
            df['estimate_tag'][i] = 'Intraday price'
        elif ~np.isnan(df['price_reg_up'][i]):
            df['estimate_intraday_price'][i] = df['price_reg_up'][i]
            df['estimate_tag'][i] = 'Up regulation price'
        elif ~np.isnan(df['price_reg_down'][i]):
            df['estimate_intraday_price'][i] = df['price_reg_down'][i]
            df['estimate_tag'][i] = 'Down regulation price'
        else:
            df['estimate_intraday_price'][i] = df['dayahead_actuals'][i]
            df['estimate_tag'][i] = 'Dayahead price'
    return df



def profit_production_df(threshold:float, df:pd.DataFrame = gather_production_df_and_calculate_estimated_intraday_prices,  path:str="models/difference_probabilities/MLP model_prod_probabilities.csv"):
    df = df.copy()

    df['datetime'] = pd.to_datetime(mlp_diff['datetime'], format='%Y-%m-%d %H:%M:%S')
    df['datetime']  = mlp_diff['datetime'] .dt.tz_convert('Europe/Berlin') + dt.timedelta(hours = +1)

    df = df[df.datetime.dt.year >= 2022]
    df['diff_price_actuals_with_estimated_intraday_price'] = df.dayahead_actuals - df.estimate_intraday_price
    #df['diff_price_forecast'] = df.dayahead_forecasts - df.intraday_forecasts
    df['profit_above_threshold'] = df['difference_probabilities'].apply(lambda x: 1 if x >= threshold else 0)
    df['hourly_profit'] = df.profit_above_threshold * df.diff_price_actuals_with_estimated_intraday_price
    df['day'] = df.datetime.dt.day
    df['week'] = df.datetime.dt.week
    df['dayofweek'] = df.datetime.dt.dayofweek
    return df




def profit_production_df_old(threshold:float, path:str="models/difference_probabilities/MLP model_prod_probabilities.csv", path_old = "data/production prob/sigmoid_differences.csv"):
    df = pd.read_csv(path)
    df_old = pd.read_csv(path_old)[['datetime', 'intraday_actuals']].rename(columns={'intraday_actuals':'intraday_actuals_interpolated'})
    df = df.merge(df_old, on='datetime')
    print(df_old)
    df.datetime = __convert_cet_datetime(df.datetime.str.split('+', expand=True).iloc[:,0])
    df = df[df.datetime.dt.year >= 2022]
    df['diff_price_actuals_with_estimated_intraday_price'] = df.dayahead_actuals - df.intraday_actuals_interpolated
    #df['diff_price_forecast'] = df.dayahead_forecasts - df.intraday_forecasts
    df['profit_above_threshold'] = df['difference_probabilities'].apply(lambda x: 1 if x >= threshold else 0)
    df['hourly_profit'] = df.profit_above_threshold * df.diff_price_actuals_with_estimated_intraday_price
    df['day'] = df.datetime.dt.day
    df['week'] = df.datetime.dt.week
    df['dayofweek'] = df.datetime.dt.dayofweek
    return df


def total_profit(threshold:float):
    df = profit_production_df(threshold)
    total_profit = df.hourly_profit.values.sum()
    return total_profit


def total_profit_big(threshold:float, groupingby:str='week'):
    df = profit_production_df(threshold)
    df.rename(inplace = True, columns={'hourly_profit':str(threshold)})
    total_profit = df[[groupingby, str(threshold)]].groupby(groupingby).sum().T
    df.rename(index={'index': 'Threshold'})
    return total_profit


def total_profit_big_old(threshold:float, groupingby:str='week'):
    df = profit_production_df_old(threshold)
    df.rename(inplace = True, columns={'hourly_profit':str(threshold)})
    total_profit = df[[groupingby, str(threshold)]].groupby(groupingby).sum().T
    df.rename(index={'index': 'Threshold'})
    return total_profit

def total_profit_matrix(thresholds:list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]):
    profit_risk_df = pd.DataFrame({'Threshold':[np.nan], 'Profit':[np.nan]})
    for t in thresholds:
        profit_risk_df = pd.concat([profit_risk_df, pd.DataFrame({"Threshold":[t], "Profit":[total_profit(t)]})])
        #profit_risk_df = pd.concat([profit_risk_df, pd.DataFrame(columns = {'Threshold':t, 'Profit':total_profit(t)})])

    profit_risk_df = profit_risk_df.dropna()
    return profit_risk_df

#active_up_reg = [1 for i in range(0, len(series_df)) if  np.isnan(series_df.iloc[i]['NO2 Price Regulation Up EUR/MWh H Actual : non-forecast']) and np.isnan(series_df.iloc[i]['NO2 Price Regulation Down EUR/MWh H Actual : non-forecast'])]
#active_down_reg = [1 for i in range(0, len(series_df)) if  np.isnan(series_df.iloc[i]['NO2 Price Regulation Down EUR/MWh H Actual : non-forecast'])]
#len(active_down_reg)

def total_profit_matrix_big(groupingby:str = 'week', thresholds:list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]):
    df = pd.DataFrame()
    for t in thresholds:
        df = pd.concat([df, total_profit_big(t, groupingby)])
    return df


def total_profit_matrix_big_old(groupingby:str = 'week', thresholds:list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]):
    df = pd.DataFrame()
    for t in thresholds:
        df = pd.concat([df, total_profit_big(t, groupingby, profit_production_df_old(t))])
    return df



if __name__ == '__main__':


    ###########################################################################################################################
    ##### Load and cleaning data ##############################################################################################
    ###########################################################################################################################

    # Load dayahead and intraday price data
    dayahead = pd.read_pickle('output_data/elspot_price_and_volume_2019_to_2022.pkl')
    intraday = pd.read_pickle('output_data/elbas_price_and_volume_2019_to_2022.pkl')


    # Assign types
    dayahead['market'] = 'Day-ahead'
    intraday['market'] = 'Intraday'


    # Concat data price data 
    nordpool = pd.concat([dayahead[['price', 'datetime', 'area', 'market', 'volume']], intraday]).reset_index(drop=True)


    # Feature engineering
    nordpool = date_features(nordpool)
    nordpool = nordpool[nordpool.datetime.dt.year >= 2019]

    # Separate Intraday and Day-ahead data
    intraday = nordpool[nordpool.market == 'Intraday']
    intraday = intraday[intraday.datetime.dt.year >= 2019] # 2019
    dayahead = nordpool[nordpool.market == 'Day-ahead']
    dayahead = dayahead[dayahead.datetime.dt.year >= 2019] # 2019


    # Feature engineering, replacing missing values with a value
    missing_tradehours_intraday         = fill_nordpool_data(df = intraday)
    missing_tradehours_intraday['hour'] = missing_tradehours_intraday['hour'].fillna(-1)
    missing_tradehours_intraday         = missing_tradehours_intraday.reset_index()
    missing_tradehours_intraday.hour    = missing_tradehours_intraday.hour.astype(int)
    missing_tradehours_intraday         = missing_tradehours_intraday.loc[:,['hour', 'area', 'index']].groupby(by = ['area','hour'], as_index = False).count()


    intraday_tradehours = fill_nordpool_data(df = intraday)
    intraday_tradehours = date_features(intraday_tradehours)
    intraday_tradehours = intraday_tradehours.fillna(value = {'volume': 0})                        # Replace trade volume nan with 0
    intraday_tradehours = intraday_tradehours.assign(hour = intraday_tradehours.datetime.dt.hour)
    intraday_tradehours = intraday_tradehours.reset_index()
    dayahead_tradehours = dayahead.copy()


    # Data frames
    nordpool_all        = pd.concat([dayahead_tradehours, intraday_tradehours]) # Concat into joined dataframe 
    intraday_all        = nordpool_all[nordpool_all.market == 'Intraday']
    dayahead_all        = nordpool_all[nordpool_all.market == 'Day-ahead']


    # Data frame Wide format
    nordpool_all_wide  = nordpool_all.pivot_table(index = ['datetime', 'area'], columns = 'market', values = 'volume').reset_index()
    nordpool_all_wide = nordpool_all_wide.dropna()


    # Calculate Intraday as share of Day-ahead
    nordpool_all_wide_area_volume_share = nordpool_all_wide.groupby(['area'], as_index = False).median()
    nordpool_all_wide_area_volume_share['share'] = nordpool_all_wide_area_volume_share['Intraday']/nordpool_all_wide_area_volume_share['Day-ahead'] 
    nordpool_all_wide_area_volume_share.sort_values(by = 'share', ascending = False, inplace = True)


    # Missing tradehours on NO2
    missing_tradehours_intraday_summary = non_traded_hours_aggregated(intraday).sort_values(by = 'non_traded_hours_percent')
    missing_tradehours_intraday_summary_hourly = intraday_tradehours.loc[intraday_tradehours.volume == 0, ['area', 'hour', 'index']].groupby(by=['area', 'hour'], as_index = False).count()


    # Median buy volume NO2
    no2 = nordpool_all[nordpool_all.area == "NO2"]
    weekly_median_volume = no2.loc[:,['market','year','weekofyear', 'volume']].groupby(by = ['market','year','weekofyear'], as_index = False).median()
    weekly_median_volume_intraday = weekly_median_volume[weekly_median_volume.market == 'Intraday'].reset_index()
    weekly_median_volume_dayahead = weekly_median_volume[weekly_median_volume.market == 'Day-ahead'].reset_index()


    # Missing tradehours between recordings
    missing_tradehours_intraday_occurances_no2 = count_missing_values_between_recordings(
        df = intraday_tradehours, 
        col = 'price',
        area = 'NO2')
    missing_tradehours_intraday_occurances_no2 = missing_tradehours_intraday_occurances_no2.groupby(by = 'hours', as_index = False).count()
    missing_tradehours_intraday_occurances_no2.columns = ['Hours', 'Frequency']
    missing_tradehours_intraday_occurances_no2.Hours = missing_tradehours_intraday_occurances_no2.Hours.astype(int)
    missing_tradehours_intraday_occurances_no2 = missing_tradehours_intraday_occurances_no2[missing_tradehours_intraday_occurances_no2.Hours != 0]
    missing_tradehours_intraday_occurances_no2 = missing_tradehours_intraday_occurances_no2.sort_values(by = 'Hours')
    missing_tradehours_intraday_occurances_no2['Market'] = 'Intraday'

    ###########################################################################################################################################
    ###### Energy Quantified Data #############################################################################################################
    ###########################################################################################################################################

    # Forecast data 
    """    forecasts_df = pd.read_pickle('data/weather/forecasts.pkl')

        # Unique forecast series
        forecast_series = forecasts_df['search_term'].unique()

        # Backcasts
        backcasts_df            = pd.read_pickle('data/weather/backcasts.pkl')
        no_backcasts            = backcasts_df[backcasts_df.zone.isin(['NO1', 'NO2', 'NO3', 'NO4', 'NO5'])]
        all_backcasts_series    = backcasts_df.search_term.unique()
        no_transmission_series  =  no_backcasts[['Exchange' in x  for x in no_backcasts.search_term]]

        # Selecting series
        hydro_precipitation = forecasts_df[forecasts_df['search_term'] == 'NO1 Hydro Precipitation Energy MWh H Forecast']
        NO1_temperature     = forecasts_df[forecasts_df['search_term'] == 'NO1 Consumption Temperature °C 15min Forecast']
        DK1_wind_production = forecasts_df[forecasts_df['search_term'] == 'DK1 Wind Power Production MWh/h 15min Forecast']"""




    ###########################################################################################################################################
    ###### Descriptive analysis INTRADAY VS DAY-AHEAD ###############################################################################################
    ###########################################################################################################################################

    ################################
    ##### Tables: Summary statistics
    ################################

    # Nord Pool prices in 2019-2022
    summary_nordpool_prices = summary_table(
        df          = nordpool.loc[:,['price', 'area', 'market']], 
        groupby     = ['market','area'])
    summary_nordpool_prices.columns = ['Market', 'Area', 'Count', 'Mean', 'St.dev', 'Min', '25%', '50%', '75%', 'Max'] 
    summary_nordpool_prices.Count = summary_nordpool_prices.Count.astype(int)
    print_latex(summary_nordpool_prices, 'Summary statistics of Nord Pool prices in 2019-2022' )




    # Nord Pool volumes in 2019-2022
    summary_nordpool_volumes = summary_table(
        df          = nordpool_all.loc[:,['market','area', 'volume']], 
        groupby     = ['market', 'area'])
    summary_nordpool_volumes.columns = ['Market', 'Area', 'Count', 'Mean', 'St.dev', 'Min', '25%', '50%', '75%', 'Max'] 
    summary_nordpool_volumes.Count = summary_nordpool_volumes.Count.astype(int)
    print_latex(summary_nordpool_volumes, 'Summary statistics of Nord Pool volumes in 2019-2022')


    # Missing tradehours NO2
    table__missing_tradehours_intraday_summary = missing_tradehours_intraday_summary.copy()
    table__missing_tradehours_intraday_summary.non_traded_hours = table__missing_tradehours_intraday_summary.non_traded_hours.astype(int)
    table__missing_tradehours_intraday_summary.non_traded_hours_percent = pd.Series(["{0:.1f}%".format(val * 100) for val in table__missing_tradehours_intraday_summary.non_traded_hours_percent ], index = missing_tradehours_intraday_summary.index)
    print_latex(table__missing_tradehours_intraday_summary, 'Number of non-traded hours and share of total number of hours')


    # Intraday volume share of day-ahead volume
    table_nord_pool_all_wide_share =   pd.Series(["{0:.3f}%".format(val * 100) for val in nordpool_all_wide_area_volume_share.share ], index = nordpool_all_wide_area_volume_share.index)
    print_latex(table_nord_pool_all_wide_share, 'Intraday volume share of day-ahead volume')


    


    ################################
    ##### Plots: Price and Volume
    ################################
    # Lineplot of Intraday vs. Day-ahead 2019-2022 for NO2
    sns_lineplot(
        df          = nordpool[nordpool.area.isin(['NO2'])], 
        x           = nordpool.datetime,
        xlab        = 'Date', 
        y           = nordpool.price, 
        ylab        = 'Price',
        hue         = 'market',
        offset      = 0, 
        lw          = 4,
        plot_title  = '',
        file_title  = 'Lineplot Intraday and day-ahead prices NO2',
        figsize     = (80,30), 
        month = True,
        save        = True)


    # Lineplot of Intraday developement 2019-2022 for NO2
    sns_lineplot(
        df          = weekly_median_volume_intraday, 
        x           = weekly_median_volume_intraday.index,
        xlab        = 'Week', 
        y           = weekly_median_volume_intraday.volume, 
        ylab        = 'Volume',
        hue         = 'year',
        offset      = 50, 
        plot_title  = '',
        file_title  = 'Lineplot Weekly median buy volume intraday market NO2',
        legend_loc = 'upper left',
        figsize = (55,35),
        lw = 10,
        bigax = True,
        save        = True)

    # Lineplot of Day-ahead developement 2019-2022 for NO2
    sns_lineplot(
        df          = weekly_median_volume_dayahead, 
        x           = weekly_median_volume_dayahead.index,
        xlab        = 'Week', 
        y           = weekly_median_volume_dayahead.volume, 
        ylab        = 'Volume',
        hue         = 'year',
        offset      = 50, 
        plot_title  = '',
        file_title  = 'Lineplot Weekly median buy volume day-ahead market NO2',
        figsize = (55,35),
        lw = 10,
        bigax = True,
        save        = True)


    # Boxplot of Intraday and day-ahead prices per hour in NO2
    sns_boxplot(
        df          = nordpool[nordpool.area == 'NO2'], 
        x           = 'hour', 
        y           = 'price',
        xlab        = 'Hour',
        ylab        = 'Price', 
        hue         = 'market',
        plot_title  = 'Hourly prices of intraday and day-ahead market NO2',
        save        = True,
        file_title  = 'Boxplot Hourly prices of intraday and day-ahead market NO2')


    # Boxplot of Intraday and day-ahead prices per weekday in NO2
    sns_boxplot(
        df          = nordpool[nordpool.area == 'NO2'], 
        x           = 'day_name', 
        y           = 'price',
        xlab        = 'Day',
        ylab        = 'Price', 
        hue         = 'market',
        plot_title  = 'Daily prices of intraday and day-ahead market NO2',
        save        = True,
        file_title  = 'Boxplot Daily prices of intraday and day-ahead market NO2')


    # Median price of intraday and day-ahead market for NO2 per hour
    sns_pointplot(
        df          = nordpool[nordpool.area == 'NO2'],  
        x           = 'hour',
        xlab        = 'Hour', 
        y           = 'price', 
        ylab        = 'Price',
        hue         = 'market', 
        plot_title  = '',  
        estimator   = np.median,
        save        = True,
        lw = 20,
        figsize = (55,15),
        file_title  = 'Pointplot Hourly median prices of intraday and day-ahead market NO2')


    # Median volume of intraday markets
    sns_lineplot(
        df          = nordpool_all[nordpool_all.market == 'Intraday'],  
        x           = 'hour',
        xlab        = 'Hour', 
        y           = 'volume', 
        ylab        = 'Volume',
        hue         = 'area', 
        plot_title  = '',  
        estimator   = np.median,
        xinterval   = 1,
        xlim_min    = 0,
        xlim_max    = 23,
        save        = True,
        figsize = (55, 25),
        lw = 10,
        bigax = True,
        file_title  = 'Lineplot Hourly median buy volume of intraday markets')


    # Median volume of intraday markets 2021
    sns_lineplot(
        df          = intraday_all[intraday_all.year == 2021],  
        x           = 'hour',
        xlab        = 'Hour', 
        y           = 'volume', 
        ylab        = 'Volume',
        hue         = 'area', 
        plot_title  = 'Hourly median buy volume of intraday markets 2021',  
        estimator   = np.median,
        xinterval   = 1,
        xlim_min    = 0,
        xlim_max    = 23,
        save        = True,
        file_title  = 'Lineplot Hourly median buy volume of intraday markets 2021')


    # Median volume of day-ahead markets
    sns_lineplot(
        df          = dayahead,  
        x           = 'hour',
        xlab        = 'Hour', 
        y           = 'volume', 
        ylab        = 'Volume',
        hue         = 'area', 
        plot_title  = 'Hourly median buy volume of day-ahead markets',  
        estimator   = np.median,
        xinterval   = 1,
        xlim_min    = 0,
        xlim_max    = 23,
        save        = True,
        file_title  = 'Lineplot Hourly median buy volume of day-ahead markets')


    # Boxplot of Intraday and day-ahead volum per hour in NO2
    sns_boxplot(
        df          = intraday_all[intraday_all.area == 'NO2'], 
        x           = 'hour', 
        y           = 'volume',
        xlab        = 'Hour',
        ylab        = 'Volume', 
        hue         = 'market',
        plot_title  = 'Hourly buy volume of intraday market NO2',
        save        = True,
        file_title  = 'Boxplot Hourly buy volume of intraday market NO2')


    # Boxplot of Intraday and day-ahead volum per hour in NO2
    sns_boxplot(
        df          = dayahead_all[dayahead_all.area == 'NO2'], 
        x           = 'hour', 
        y           = 'volume',
        xlab        = 'Hour',
        ylab        = 'Volume', 
        hue         = 'market',
        plot_title  = 'Hourly buy volume of day-ahead market NO2',
        save        = True,
        file_title  = 'Boxplot Hourly buy volume of day-ahead market NO2')


    # Histplot of missing tradehours on NO2
    sns_barplot(
        df          = missing_tradehours_intraday[missing_tradehours_intraday.area == 'NO2'], 
        x           = 'hour',
        xlab        = 'Hour',
        y           = 'index', 
        ylab        = 'Frequency',
        hue         = 'area',
        plot_title  = 'Tradehours frequency intraday market NO2',
        file_title  = 'Barplot Tradehours frequency intraday market NO2',
        save        = True)



    # Histplot of missing tradehours on NO2
    sns_barplot(
        df          = missing_tradehours_intraday_summary_hourly[missing_tradehours_intraday_summary_hourly.area == 'NO2'], 
        x           = 'hour',
        xlab        = 'Hour',
        y           = 'index', 
        ylab        = 'Frequency',
        hue         = 'area',
        plot_title  = 'Hourly missing tradehours intraday market NO2',
        file_title  = 'Barplot Hourly missing tradehours intraday market NO2',
        save        = True)


    # Histplot of missing tradehours intraday
    sns_barplot(
        df          = missing_tradehours_intraday_summary_hourly, 
        x           = 'hour',
        xlab        = 'Hour',
        y           = 'index', 
        ylab        = 'Frequency',
        hue         = 'area',
        plot_title  = 'Hourly missing tradehours intraday markets',
        file_title  = 'Barplot Hourly missing tradehours intraday markets',
        save        = True)


    # Histplot of missing tradehours intraday
    sns_barplot(
        df          = missing_tradehours_intraday_occurances_no2, 
        x           = 'Hours',
        xlab        = 'Number of hours',
        y           = 'Frequency', 
        ylab        = 'Frequency',
        hue         = 'Market',
        plot_title  = '',
        file_title  = 'Barplot Number of missing tradehours between traded hours on intraday market NO2',
        figsize = (55,25),
        save        = True)

    ################################
    ##### Other
    ################################



    # Add lag to series by widening the dataframe, appending the lag, and the elongating the dataframe
    wide_nordpool = long_to_wide_df(nordpool.copy())
    wide_nordpool = generate_time_lag(wide_nordpool,  'Day-ahead', 1)
    nordpool = wide_to_long_df(wide_nordpool)


    # Intraday vs Day-ahead lag 1 for NO2
    dayahead_intraday_plot_data = nordpool[(nordpool['area'] == 'NO2') & (nordpool['market'] != 'dayahead_lag1') & (nordpool['datetime'].dt.year > 2020)]
    sns_lineplot(df = dayahead_intraday_plot_data, x = 'datetime', y = 'price', hue = 'market')



    ## Intraday and Day-ahead cross-correlations
    #TODO fix this
    intraday_dayahead_corr_df = cross_correlations(wide_nordpool, 'Day-ahead', 'dayahead_lag1', 20) # Does not work as of 21-02
    sns_lineplot(df = intraday_dayahead_corr_df, x = 'lags', y = 'correlation')
    sns_barplot(df = intraday_dayahead_corr_df, x = 'lags', y = 'correlation')



    ###########################################################################################################################################
    ###### Analysis of which hours are traded on Intraday ########################################################################################
    ###########################################################################################################################################

    # Volume and trade hour data for Day-ahead
    dayahead_tradehours = dayahead.copy()
    nordpool_all = pd.concat([dayahead_tradehours, intraday_tradehours]) # Concat into joined dataframe 
    no2_nordpool_all_tradehours = nordpool_all[nordpool_all['area'] == 'NO2']
    no2_volume_aggregated = no2_nordpool_all_tradehours.groupby(['hour', 'market'], as_index=False).mean('volume')
    no2_volume_aggregated_melted = no2_volume_aggregated.melt(id_vars = 'hour', var_name = 'market', value_vars = ['price', 'volume'])
    sns_barplot(df = no2_volume_aggregated, x = 'hour', y = 'volume', hue = 'market')







    ###########################################################################################################################################
    ###### Descriptive analysis independent variables #########################################################################################
    ###########################################################################################################################################

    # Plotting forecasts
    line_plot(df = NO1_temperature,     x = 'date', y = 'value')
    line_plot(df = hydro_precipitation, x = 'date', y = 'value')
    line_plot(df = DK1_wind_production, x = 'date', y = 'value')



    ###########################################################################################################################################
    ###### Time Series decomposition ##########################################################################################################
    ###########################################################################################################################################
    no2_intraday = no2[no2.market == 'Intraday']
    no2_dayahead = no2[no2.market == 'Day-ahead']

    no2_intraday = remove_nans(no2_intraday, 'linear')
    #no2_intraday.index = pd.DatetimeIndex(data = no2_intraday.datetime) #, freq = 'h')
    no2_intraday = no2_intraday.loc[:,['price']]


    
    no2_dayahead = remove_nans(no2_dayahead, 'linear')
    #no2_dayahead.index = pd.DatetimeIndex(data = no2_dayahead.datetime) #, freq = 'h')
    no2_dayahead = no2_dayahead.loc[:,['price']]

    # Multiplicative Decomposition 
    #result_mul = seasonal_decompose(no2_intraday['price'], model='multiplicative', extrapolate_trend='freq')

    # Additive Decomposition
    result_add_intraday = seasonal_decompose(no2_intraday['price'], model='additive', extrapolate_trend='freq')
    result_add_dayahead = seasonal_decompose(no2_dayahead['price'], model='additive', extrapolate_trend='freq')


    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add_intraday.plot().suptitle('Additive Decompose of intraday price NO2', fontsize=22)
    result_add_dayahead.plot().suptitle('Additive Decompose of dayahead price NO2', fontsize=22)
    plt.show()




    # ADF Test
    check_stationarity(remove_nans(series_df, 'linear')['dayahead_price'], testtype = 'adfuller')
    check_stationarity(no2_intraday.price, testtype = 'adfuller')
    

    # KPSS Test
    check_stationarity(no2_dayahead.price, testtype = 'kpss')
    check_stationarity(no2_intraday.price, testtype = 'kpss')


    # ACF and PACF for DAYAHEAD
    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
    plot_acf(no2_dayahead.price.tolist(), lags=50, ax=axes[0], alpha = 0.05)
    plot_pacf(no2_dayahead.price.tolist(), lags=50, ax=axes[1], alpha = 0.05)

    # ACF and PACF for DAYAHEAD
    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
    plot_acf(no2_intraday.price.tolist(), lags=50, ax=axes[0], alpha = 0.05)
    plot_pacf(no2_intraday.price.tolist(), lags=50, ax=axes[1], alpha = 0.05)


    def plot_acf_pacf(series, ndiff:int = 0):
        series = series.copy().diff(periods = ndiff)
        series = series.dropna()
        print(series)
        fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
        sm.graphics.tsa.plot_acf(series.values.tolist(), lags=50, ax=axes[0], alpha = 0.05)
        sm.graphics.tsa.plot_pacf(series.values.tolist(), lags=50, ax=axes[1], alpha = 0.05)
    
    plot_acf_pacf(no2_intraday.price, ndiff = 1)

    from pmdarima.arima import auto_arima
    model = auto_arima(no2_intraday.price, start_p=0, start_q=0)
    model.summary()


    ### Calculate mean dayahead and intraday price
    cv = get_cv_type('sliding_evaluation', 5, **{'val_size':114})
    t  = [t for t in cv.split(series_df)]

    test_indices = [t[i][2] for i in range(0, len(t))]

    test_indices_list =  result = sum(test_indices, [])

    series_df.iloc[test_indices_list].intraday_price.mean()
    series_df.iloc[test_indices_list].dayahead_price.mean()








    ###########################################################################################################################################
    ###### APPENDIX of different plots (nice to have) #########################################################################################
    ###########################################################################################################################################


    # Barplot of intraday and dayahead(Weekdays)
    sns_barplot(df = intraday, x = 'day_name', y = 'price', hue = 'area', plot_title = 'Intraday mean price per day 2019-2022',    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    sns_barplot(df = dayahead, x = 'day_name', y = 'price', hue = 'area', plot_title = 'Day-ahead mean price per day 2019-2022',   order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    sns_barplot(df = intraday, x = 'day_name', y = 'price', hue = 'area', plot_title = 'Intraday median price per day 2019-2022',  order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], estimator=np.median)
    sns_barplot(df = dayahead, x = 'day_name', y = 'price', hue = 'area', plot_title = 'Day-ahead median price per day 2019-2022', order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], estimator=np.median)


    # Barplot of intraday and dayahead (Hour)
    sns_barplot(df = intraday, x = 'hour', y = 'price', hue = 'area', plot_title = 'Intraday mean price per hour 2019-2022')
    sns_barplot(df = dayahead, x = 'hour', y = 'price', hue = 'area', plot_title = 'Day-ahead mean price per hour 2019-2022')
    sns_barplot(df = intraday, x = 'hour', y = 'price', hue = 'area', plot_title = 'Intraday median price per hour 2019-2022',  estimator=np.median)
    sns_barplot(df = dayahead, x = 'hour', y = 'price', hue = 'area', plot_title = 'Day-ahead median price per hour 2019-2022', estimator=np.median)


    # Histplot of intraday and dayahead prices
    sns_histplot(df = intraday, x = 'price', hue = 'area', plot_title = 'Histogram of intraday prices')
    sns_histplot(df = dayahead, x = 'price', hue = 'area', plot_title = 'Histogram of day-ahead prices')

    # Histplot of intraday aggregated hourly trading volume
    sns_histplot(df = intraday, x = 'volume', hue = 'area', plot_title = 'Histogram of intraday volume')

    # Median Lineplot of prices per hour and day
    sns_lineplot(df = intraday, x = 'hour', y = 'price', hue = 'area', plot_title='Median intraday price per hour')
    sns_lineplot(df = dayahead, x = 'hour', y = 'price', hue = 'area', plot_title='Median day-ahead price per hour')
    sns_lineplot(df = intraday, x = 'weekday', y = 'price', hue = 'area', plot_title='Median intraday price per day')
    sns_lineplot(df = dayahead, x = 'weekday', y = 'price', hue = 'area', plot_title='Median day-ahead price per day')

    # Lineplot of Intraday vs. Day-ahead 2019-2022 for NO2
    sns_lineplot(
        df = intraday_all, 
        x = nordpool.datetime,
        xlab = 'Date', 
        y = intraday_all.volume, 
        ylab = 'Volume',
        hue = 'market',
        offset = 50,
        plot_title = 'Intraday vs. Day-ahead prices',
        file_title = 'Intraday vs Day-ahead NO2',
        save = False)






    
    


    # Table of missing records for each feature
    na_df = pd.DataFrame(series_df.isna().sum()).reset_index()
    na_df.columns = ['Series', "Frequency of missing values"]
    na_df = na_df[na_df["Series"].str.contains("lag")==False].sort_values(['Frequency of missing values','Series'], ascending=False)
    na_df['Share of total observations'] = na_df["Frequency of missing values"]/len(series_df)
    na_df['Share of total observations'] = pd.Series(["{0:.2f}%".format(val * 100) for val in na_df['Share of total observations']], index = na_df.index)
    print_latex(na_df, 'Summary of missing data')

    # Histogram of numerical features
    series_without_lags_df = series_df.loc[:,series_df.columns.str.contains("lag")==False]
    series_without_lags_df = series_without_lags_df.select_dtypes(["float64", "int64"])

    n_cols = 4
    n_series = len(series_without_lags_df.columns)
    bins = 30
    n_subplots = 2
    for j in range(0,2):
        fig, axs = plt.subplots(math.ceil(n_series/n_cols/n_subplots), n_cols, tight_layout=False, figsize=(60,80)) # figsize=(60,120) best config
        for i, serie in enumerate(series_without_lags_df.iloc[:,j*(n_series//n_subplots):(j+1)*(n_series//n_subplots)]):            
            col = i % n_cols
            row = i // n_cols
            axs[row, col].hist(series_without_lags_df.loc[:,serie], bins=bins, ec = "k", color = sns.color_palette('Set2')[2])
            axs[row, col].set_title(serie, fontsize = 24)
        plt.savefig('../static/histograms ' + str(j+1) + ' av ' + str(n_subplots) + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')


    # Detecting outliers box-plot
    series_without_lags_df['hour'] = series_df.hour.astype('int64')
    sns.set_style(sns_theme)
    sns.set_style({'font.family':'serif', 'font.serif':font})
    n_cols = 4
    n_series = len(series_without_lags_df.columns)
    n_subplots = 2
    for j in range(0,2):
        fig, axs = plt.subplots(math.ceil(n_series/n_cols/n_subplots), n_cols, tight_layout=False, figsize=(60,80)) # figsize=(60,120) best config
        for i, serie in enumerate(series_without_lags_df.iloc[:,j*(n_series//n_subplots):(j+1)*(n_series//n_subplots)]):
            col = i % n_cols
            row = i // n_cols
            sns.boxplot(
                y=series_without_lags_df.loc[:,serie], 
                x = series_without_lags_df.hour, 
                ax = axs[row, col], 
                color = sns.color_palette('Set2')[0])
            axs[row, col].set_title(serie, fontsize = 24)
        plt.savefig('../static/boxplots ' + str(j+1) + ' av ' + str(n_subplots) + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')


    # Neptune analysis
    best_models_df = pd.read_excel("data/excel/neptune_logging.xlsx", sheet_name="Best models")
    best_models_df = best_models_df.iloc[:,2:]
    best_models_df = best_models_df[best_models_df.Architecture.isin(['GRU', 'TFT', 'LSTM', 'DeepAR'])]
    print_latex(best_models_df, 'Hyperparameter selection using Nested Cross-Validation', float_format = '%.3f')


    # Neptune analysis
    #best_benchmark_models_df = pd.read_excel("../data/excel/neptune_logging.xlsx", sheet_name="Best benchmark models")
    best_models_df = pd.read_excel("data/excel/neptune_logging.xlsx", sheet_name="Best models")
    best_models_df = best_models_df.loc[:,~best_models_df.columns.isin(['Id', 'Creation Time', 'Hidden Size',
       'Learningrate', 'RNN layers', 'ROPR', 'ROPP'])]
    best_benchmark_models_df = best_models_df[~best_models_df.Architecture.isin(['GRU', 'TFT', 'LSTM', 'DeepAR'])]
    print_latex(best_benchmark_models_df, 'Hyperparameter selection using Nested Cross-Validation', float_format = '%.3f')

    # Neptune analysis
    best_performing_model_df = pd.read_excel("data/excel/neptune_logging.xlsx", sheet_name="Best performing model")
    best_performing_model_df = best_performing_model_df.iloc[:,2:].sort_values(by=['Target','Test MAE'])
    best_performing_model_df = best_performing_model_df.loc[:,['Architecture','Target','Test SMAPE','Test MAE','Test RMSE','Train SMAPE', 'Train MAE', 'Train RMSE']]
    print_latex(best_performing_model_df, 'Model performance using Nested Cross-Validation', float_format = '%.3f')

    # Neptune analysis
    best_performing_benchmark_model_df = pd.read_excel("data/excel/neptune_logging.xlsx", sheet_name="Best performing benchmark model")
    best_performing_benchmark_model_df = best_performing_benchmark_model_df.iloc[:,2:].sort_values(by=['Target','Test MAE'])
    best_performing_benchmark_model_df = best_performing_benchmark_model_df.loc[:,['Architecture','Target','Test SMAPE','Test MAE','Test RMSE','Train SMAPE', 'Train MAE', 'Train RMSE']]
    print_latex(best_performing_benchmark_model_df, 'Model performance using Nested Cross-Validation', float_format = '%.3f')


    # Neptune best diff model
    best_diff_model_df = pd.read_excel("data/excel/neptune_logging.xlsx", sheet_name="Best diff net")
    best_diff_model_df = best_diff_model_df.iloc[:,2:].sort_values(by=['Test ROC AUC'], ascending = False)
    print_latex(best_diff_model_df, 'Best diff network', float_format = '%.3f')



    # OLD OLD OLD OLD
    profit_matrix_day_old = total_profit_matrix_big_old('day').T.reset_index()
    profit_matrix_day_old = pd.concat([profit_matrix_day_old, pd.DataFrame(profit_matrix_day_old.sum()).T])
    profit_matrix_day_old = profit_matrix_day_old.loc[:,profit_matrix_day_old.columns != '0']



    # Run these in order to get profit by prob group by date
    profit_matrix_week = total_profit_matrix_big('week').reset_index()
    profit_matrix_dayofweek = total_profit_matrix_big('dayofweek')
    profit_matrix_day = total_profit_matrix_big('day').T.reset_index()
    profit_matrix_day = pd.concat([profit_matrix_day, pd.DataFrame(profit_matrix_day.sum()).T])
    profit_matrix_day = profit_matrix_day.loc[:,profit_matrix_day.columns != '0']

    print_latex(profit_matrix_week, 'Weekly profits', float_format = '%.2f')
    print_latex(profit_matrix_dayofweek, 'Day of week profits', float_format = '%.2f')
    print_latex(profit_matrix_day, 'Daily profits', float_format = '%.1f')







    # Merit order curve
    merit_order = pd.read_csv("../data/meritorder_norway.csv", skiprows=4)
    merit_order.rename(columns = {'Unnamed: 0':'year'}, inplace = True)
    merit_order_2020_norway = merit_order[merit_order['year']==2020]
    merit_order_2020_norway.melt(id_vars = ['year', 'Units']).sort_values('value')
    merit_order_2020_norway




    figsize = (50,45)
    fontsize = 45
    # Hourly MAE per cv fold
    for i in range(1,6):
        df = mean_mae_per_hour(retrieve_predictions_cv_fold("data/cv_data/intraday/fold"+str(i)))
        df = df.replace('fold [0-9]', '', regex=True)
        
        sns_lineplot(
            df=df, 
            x='hour', 
            y='HMAE', 
            xlab='Hours', 
            ylab='MAE', 
            hue='filename', 
            plot_title='Hourly Performance MAE on Intraday Fold ' + str(i), 
            file_title='intraday_hourly_cv_mae_fold'+str(i), 
            figsize=figsize,
            title_fontsize=fontsize,
            lw = 6,
            save=True)

    # Horizon index MAE per cv fold
    for i in range(1,6):
        df = mean_mae_per_horizon_index(retrieve_predictions_cv_fold("data/cv_data/intraday/fold"+str(i)))
        df = df.replace('fold [0-9]', '', regex=True)
        sns_lineplot(
            df=df, 
            x='horizon_index', 
            y='HMAE', 
            xlab='Hours', 
            ylab='MAE', 
            hue='filename', 
            plot_title='Evaluation MAE across forecast horizon on Intraday Fold ' + str(i), 
            file_title='intraday_horizon_cv_mae_fold'+str(i), 
            title_fontsize=fontsize,
            figsize=figsize,
            save=True)

    # Hourly MAE per cv fold
    for i in range(1,6):
        df = mean_mae_per_hour(retrieve_predictions_cv_fold("data/cv_data/dayahead/fold"+str(i)))
        df = df.replace('fold [0-9]', '', regex=True)
        sns_lineplot(
            df=df, 
            x='hour', 
            y='HMAE', 
            xlab='Hours', 
            ylab='MAE', 
            hue='filename', 
            plot_title='Hourly Performance MAE on Day-ahead Fold ' + str(i), 
            file_title='dayahead_hourly_cv_mae_fold'+str(i), 
            title_fontsize=fontsize,
            figsize=figsize,
            save=True)

    # Horizon index MAE per cv fold
    for i in range(1,6):
        df = mean_mae_per_horizon_index(retrieve_predictions_cv_fold("data/cv_data/dayahead/fold"+str(i)))
        df = df.replace('fold [0-9]', '', regex=True)
        sns_lineplot(
            df=df, 
            x='horizon_index', 
            y='HMAE', 
            xlab='Hours', 
            ylab='MAE', 
            hue='filename', 
            plot_title='Evaluation MAE across forecast horizon on Day-ahead Fold ' + str(i), 
            file_title='dayahead_horizon_cv_mae_fold'+str(i), 
            title_fontsize=fontsize,
            figsize=figsize,
            save=True)






    figsize = (55,35)
    legend_loc = "upper left"
    lw = 10

    # Hourly MAE per cv fold
    
    df = mean_mae_per_hour(retrieve_predictions_cv_all_folds("data/cv_data/dayahead/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='hour', 
        y='HMAE', 
        xlab='Hour', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Hour and horizon mae for all folds/dayahead_hourly_cv_mae_all_folds', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = legend_loc,
        lw = lw,
        bigax = True,
        save=True)

    # Horizon index MAE per cv fold
    df = mean_mae_per_horizon_index(retrieve_predictions_cv_all_folds("data/cv_data/dayahead/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='horizon_index', 
        y='HMAE', 
        xlab='Hour', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Hour and horizon mae for all folds/dayahead_horizon_cv_mae_all_folds', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = legend_loc,
        lw = lw,
        bigax = True,
        save=True)


        # Hourly MAE per cv fold
    
    df = mean_mae_per_hour(retrieve_predictions_cv_all_folds("data/cv_data/intraday/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='hour', 
        y='HMAE', 
        xlab='Hour', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Hour and horizon mae for all folds/intraday_hourly_cv_mae_all_folds', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = legend_loc,
        lw = lw,
        bigax = True,
        save=True)

    # Horizon index MAE per cv fold
    df = mean_mae_per_horizon_index(retrieve_predictions_cv_all_folds("data/cv_data/intraday/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='horizon_index', 
        y='HMAE', 
        xlab='Hour', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Hour and horizon mae for all folds/intraday_horizon_cv_mae_all_folds', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = legend_loc,
        lw = lw,
        bigax = True,
        save=True)







    figsize = (55,35)
    legend_loc = "upper left"

    # Horizon index MAE per cv fold
    df = mean_mae_per_fold(retrieve_predictions_cv_all_folds("data/cv_data/intraday/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='fold', 
        y='FMAE', 
        xlab='Fold', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Fold/intraday_fold_cv_mae', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = "upper left",
        lw = lw,
        bigax = True,
        save=True)

    
    
        # Horizon index MAE per cv fold
    df = mean_mae_per_fold(retrieve_predictions_cv_all_folds("data/cv_data/dayahead/all"))
    df = df[~df.filename.isin(['Mean'])]
    sns_lineplot(
        df=df, 
        x='fold', 
        y='FMAE', 
        xlab='Fold', 
        ylab='MAE', 
        hue='filename', 
        plot_title='', 
        file_title='Fold/dayahead_fold_cv_mae', 
        title_fontsize=fontsize,
        figsize=figsize,
        legend_loc = "upper left",
        lw = lw,
        bigax = True,
        save=True)



    import neptune.new as neptune

    run = neptune.init(
        project='MasterThesis/performance',
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        run='PER-169', # for example 'SAN-123'
        mode="read-only")
    
    for i in range(1,6):
        run["forecasts/fold"+str(i)].download("data/production")

    run.stop()