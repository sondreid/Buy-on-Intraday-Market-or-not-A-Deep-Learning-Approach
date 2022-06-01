""""
Script for supporting functions for model creation, training and evaluation


"""


from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split, TimeSeriesSplit, cross_val_score
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from fractions import Fraction
from decimal import Decimal
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, normalize
from termcolor import colored
import copy







class BlockingTimeSeriesSplit():

    """

    Class for blocking time series, i.e
    timeseries-based data splits which are non-overlapping

    Heavily inspired from  https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/ 
    
    """
    def __init__(self, n_splits, train_test_ratio = 0.8, random_state = 123, **kwargs):
        """
        Class constructor
        Sets new parameters based on the kwargs arguments
        """
        self.n_splits  = n_splits
        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_test_ratio
        self.random_state = random_state
        if kwargs.get('train_test_ratio') is not None:
            self.train_test_ratio = kwargs.get('train_test_ratio')

        if kwargs.get('random_state') is not None:
            self.random_state = kwargs.get('random_state')
        if kwargs.get('train_val_ratio') is not None:
            self.train_val_ratio = kwargs.get('train_val_ratio')
    
    def get_n_splits(self):
        return self.n_splits

    def get_train_test_ratio(self):
        return self.train_test_ratio


    def get_type(self) -> str:
        """
        Returns a string name of the type of cross validation the class performs
        """
        return "Blocking"

    def split(self, X):
        """
        Public method
        Sets window size based on the size of an input dataframe/numpy array.
        Returns an iterator object for train, val and test indices
        """
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_test_ratio * (stop - start)) + start
            train, val = train_test_split(indices[start: mid], train_size= self.train_val_ratio, shuffle= False, random_state= self.random_state)
            yield train, val, indices[mid + margin: stop]


class SlidingWindowSplitThreefolds():
    """
    
    Heavily inspired from:
    https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/

    Class contains implementation of Sliding window cross validation for time series.
    For all indices, several folds are drawn, for each iteration "sliding" accross the dataset.
    Increasing the last index of the fold, while increasing the starting index, keeping fold size 
    constant.

    NOTE: 
    There is non-intended behaviour that makes the number of specified splits and
    actual splits deviate. Testing as of 28.02 suggests this is an irratiting, but non-critical
    fault
    """



    def __init__(self, n_splits, train_test_ratio = 0.8, random_state = 123,  **kwargs):
        """
        Class constructor
        Sets new parameters based on the kwargs arguments
        """
        self.n_splits  = n_splits
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        
        print("Kwargs {}".format(kwargs))

        if kwargs.get('train_test_ratio') is not None:
            self.train_test_ratio = kwargs.get('train_test_ratio')

        if kwargs.get('random_state') is not None:
            self.random_state = kwargs.get('random_state')

        self.train_val_ratio = kwargs.get('train_val_ratio')
        if  self.train_val_ratio  is  None:
            self.train_val_ratio = train_test_ratio
        print("> Train val ratio", self.train_val_ratio)
        
    def split(self, X):
        """
        Public method
        Sets window size based on the size of an input dataframe/numpy array.
        Returns an iterator object for train, val and test indices
        """
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits+22
        test_ratio = round(1 - self.train_test_ratio, 2)
        frac =  Decimal(str(test_ratio))
        test_splits, train_splits  = frac.as_integer_ratio()
        print(colored("> Test splits {} : Train splits {}".format(test_splits, train_splits),  "green"))
       
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) <= 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        for i, test_start in zip(range(len(test_starts)),
                                    test_starts):
            rem = 0
            if i == 0:
                rem = n_samples % n_folds
            train,val  = train_test_split(indices[(test_start - train_size - rem):test_start], train_size =  self.train_val_ratio, shuffle= False, random_state=self.random_state)
            yield (train, val, indices[test_start:test_start + test_size])
    
    
    def get_n_splits(self):
        return self.n_splits


    def get_type(self) -> str:
        """
        Returns a string name of the type of cross validation the class performs
        """
        return "Sliding"

class SlidingWindowSplit():
    """
    
    Heavily inspired from:
    https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/

    Class contains implementation of Sliding window cross validation for time series.
    For all indices, several folds are drawn, for each iteration "sliding" accross the dataset.
    Increasing the last index of the fold, while increasing the starting index, keeping fold size 
    constant.

    NOTE: 
    There is non-intended behaviour that makes the number of specified splits and
    actual splits deviate. Testing as of 28.02 suggests this is an irratiting, but non-critical
    fault
    """



    def __init__(self, n_splits, train_test_ratio = 0.9, random_state = 123,  **kwargs):
        """
        Class constructor
        Sets new parameters based on the kwargs arguments
        """
        self.n_splits  = n_splits
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        

        if kwargs.get('train_test_ratio') is not None:
            self.train_test_ratio = kwargs.get('train_test_ratio')

        if kwargs.get('random_state') is not None:
            self.random_state = kwargs.get('random_state')

        self.train_val_ratio = kwargs.get('train_val_ratio')
        if  self.train_val_ratio  is  None:
            self.train_val_ratio = train_test_ratio
        
        self.val_size = kwargs.get('val_size')

        #print("> Train val ratio", self.train_val_ratio)
        
    def split(self, X):
        """
        Public method
        Sets window size based on the size of an input dataframe/numpy array.
        Returns an iterator object for train, val and test indices
        """
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits+10
        test_ratio = round(1 - self.train_test_ratio, 2)
        frac =  Decimal(str(test_ratio))
        test_splits, train_splits  = frac.as_integer_ratio()
        #print(colored("> Test splits {} : Train splits {}".format(test_splits, train_splits),  "green"))
        n_folds = n_splits+train_splits+test_splits -1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) <= 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        for i, test_start in zip(range(len(test_starts)),
                                    test_starts):
            rem = 0
            if i == 0:
                rem = n_samples % n_folds
            if self.val_size is not None:
                train,val  = train_test_split(indices[(test_start - train_size - rem):test_start], test_size =  self.val_size, shuffle= False, random_state=self.random_state)
            else:
                train,val  = train_test_split(indices[(test_start - train_size - rem):test_start], train_size =  self.train_val_ratio, shuffle= False, random_state=self.random_state)
            yield (train.tolist(), val.tolist(), indices[test_start:test_start + test_size].tolist())
    
    
    def get_n_splits(self):
        return self.n_splits


    def get_type(self) -> str:
        """
        Returns a string name of the type of cross validation the class performs
        """
        return "Sliding"



class SlidingWindowSplitEvaluation():
    """
    
    Heavily inspired from:
    https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/

    Class contains implementation of Sliding window cross validation for time series.
    For all indices, several folds are drawn, for each iteration "sliding" accross the dataset.
    Increasing the last index of the fold, while increasing the starting index, keeping fold size 
    constant.

    NOTE: 
    There is non-intended behaviour that makes the number of specified splits and
    actual splits deviate. Testing as of 28.02 suggests this is an irratiting, but non-critical
    fault
    """



    def __init__(self, n_splits, train_test_ratio = 0.8, random_state = 123,  **kwargs):
        """
        Class constructor
        Sets new parameters based on the kwargs arguments
        """
        self.n_splits  = n_splits
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        

        if kwargs.get('train_test_ratio') is not None:
            self.train_test_ratio = kwargs.get('train_test_ratio')

        if kwargs.get('random_state') is not None:
            self.random_state = kwargs.get('random_state')

        self.train_val_ratio = kwargs.get('train_val_ratio')
        if  self.train_val_ratio  is  None:
            self.train_val_ratio = train_test_ratio
        
        self.val_size = kwargs.get('val_size')

        #print("> Train val ratio", self.train_val_ratio)
        
    def split(self, X):
        """
        Public method
        Sets window size based on the size of an input dataframe/numpy array.
        Returns an iterator object for train, val and test indices
        """
        n_samples = len(X)
        n_splits = self.n_splits
        
        test_ratio = round(1 - self.train_test_ratio, 2)
        frac =  Decimal(str(test_ratio))
        test_splits, train_splits  = frac.as_integer_ratio()
        n_folds = n_splits+train_splits+test_splits -1
        print(colored("> Test splits {} : Train splits {}".format(test_splits, train_splits),  "green"))
       
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) <= 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        
        gap = int(0.8* test_size)
        for i, test_start in zip(range(len(test_starts)),
                                    test_starts):

            
            rem = 0
            end_gap = 0
            if i == 0:
                rem = n_samples % n_folds
                train_gap = 0
            elif i == self.n_splits-1:
                train_gap = copy.deepcopy(gap)
                end_gap = copy.deepcopy(gap) 
                print("ENDGAP")
            else: train_gap = copy.deepcopy(gap) 
                
            if self.val_size is not None:
                train,val  = train_test_split(indices[(test_start - train_size - rem + train_gap-end_gap):test_start+end_gap], test_size =  self.val_size, shuffle= False, random_state=self.random_state)    
            else:
                train,val  = train_test_split(indices[(test_start - train_size - rem + train_gap):test_start+end_gap], train_size =  self.train_val_ratio, shuffle= False, random_state=self.random_state)
            yield (train.tolist(), val.tolist(), indices[test_start+end_gap:test_start + test_size -gap +end_gap].tolist())
    
    
    def get_n_splits(self):
        return self.n_splits


    def get_type(self) -> str:
        """
        Returns a string name of the type of cross validation the class performs
        """
        return "SlidingWindowSplitEvaluation"





class ExpandingWindowSplit():
    """
    Wrapper class for sklearn TimeSeriesSplit().
    Performs expanding window cross validaiton for timeseries, i.e, 
    an overlapping and increasing fold sizes from starting index to last index.
    """

    def __init__(self, n_splits, gap = 0, train_test_ratio = 0.8, random_state = 123,  **kwargs):
        """
        Class constructor
        Sets new parameters based on the kwargs arguments
        """
        self.n_splits = n_splits
        self.gap = gap
        self.random_state = random_state
        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_test_ratio
        if kwargs.get('gap') is not None:
            self.gap = kwargs.get('gap')

        if kwargs.get('train_test_ratio') is not None:
            self.train_test_ratio = kwargs.get('train_test_ratio')
        
        if kwargs.get('random_state') is not None:
            self.random_state = kwargs.get('random_state')
        
        if kwargs.get('train_val_ratio') is not None:
            self.train_val_ratio = kwargs.get('train_val_ratio')

        self.expandingWindowSplit = TimeSeriesSplit(n_splits=n_splits, gap=self.gap)

    def get_n_splits(self):
        return self.n_splits

    def get_train_test_ratio(self):
        return self.train_test_ratio



    def __split_train_val(self, X):
        """
        Private method
        Adds a validation sets and joins them in a tuple with training and test sets
        in the same format as the native sklearn.TimeSeriesSplit.split()
        """
        train_val_test_indices = []
        for train, test in self.expandingWindowSplit.split(X):
            train, val = train_test_split(train, test_size= (1-self.train_val_ratio), shuffle = False, random_state=self.random_state)
            train_val_test_indices.append((train, val, test))

        return train_val_test_indices

    def __calculate_test_size(self, X):
        """
        Private method
        Test size should be such that the toal
        """
        test_size = len(X)*(1-self.train_test_ratio)
        test_fold_size = int(test_size/self.n_splits)
        return test_fold_size


    def split(self, X):
        """
        Public method
        Sets window size based on the size of an input dataframe/numpy array.
        Returns an iterator object for train, val and test indices
        """
        self.expandingWindowSplit.test_size = self.__calculate_test_size(X)
        train_val_test_indices = self.__split_train_val(X)
        return train_val_test_indices

    def get_type(self) -> str:
        """
        Returns a string name of the type of cross validation the class performs
        """
        return "Expanding"

def __test_cv___(num_samples, cv):
    """
    Private method for utilities script that tests a given 
    CV method that follows the predetermined CV parameter format.
    """
    X = range(0,num_samples)
    for train_indices, val_indices, test_indices in cv.split(X):
        print("> Training indices {} {}".format(train_indices[0], train_indices[-1]))
        print("> Val indices {} {}".format(val_indices[0], val_indices[-1]))
        print("> Test indices {} {}".format(test_indices[0], test_indices[-1]))

    t = [t for t in cv.split(X)]
    print("> Numnber of folds", len(t))
    return t
    


def __test_cv_two_folds__(num_samples, cv):
    """
    Private method for utilities script that tests a given 
    CV method that follows the predetermined CV parameter format.
    """
    X = range(0,num_samples)
    for train_indices, test_indices in cv.split(X):
        print("> Training indices", train_indices)
        print("> Test indices", test_indices)

    t = [t for t in cv.split(X)]
    print("> Number of folds", len(t))
    return t
    





def get_cv_type(cv_type, n_splits, **kwargs):
    """
    Returns a cv class corresponding to input cross validation types
    Per now blocking and overlapping cross validaiton inplemented
    """

    cv_types = {
        "blocking": BlockingTimeSeriesSplit,
        "expanding": ExpandingWindowSplit,
        "sliding_three_folds": SlidingWindowSplitThreefolds,
        "sliding_evaluation": SlidingWindowSplitEvaluation,
        "sliding": SlidingWindowSplit,
    }
    return cv_types.get(cv_type.lower())(n_splits, **kwargs)




def train__val_test_split(df, target, train_ratio):
    """
    Splits into train, validation and test sets and outputs seperate target and feature
    """
    y = df[[target]]
    X = df.drop(columns=[target])

    X_train, X_val_test, y_train, y_val_test = train_test_split(X,
                                                                y,  
                                                                test_size=(1-train_ratio), 
                                                                shuffle=False,  # For obvious reasons
                                                                random_state=123)

    X_val, X_test, y_val, y_test = train_test_split(X_val_test, 
                                                    y_val_test,     
                                                    test_size= 0.5, 
                                                    shuffle=False, # For obvious reasons
                                                    random_state=123)      
    return X_train, X_test, X_val, y_val,  y_train, y_test


def train_val_test_split_joined(df, train_ratio):
    """
    Splits into train, validation and test sets and outputs a dataset with target and features
    """

    train_df, val_test_df, = train_test_split(df,
                                              test_size=(1-train_ratio), 
                                              shuffle=False,  # For obvious reasons
                                              random_state=123)

    val_df, test_df, = train_test_split(val_test_df,
                                        test_size=0.5, 
                                        shuffle=False,  # For obvious reasons
                                        random_state=123)
    return train_df,val_df, test_df



def ordinal_encoding(X_train, X_test):
    oe = preprocessing.OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


def generate_time_lags(df:pd.DataFrame, variable:str, n_lags:int):
    """
    Function which lagas
    """
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"{variable}_lag{n}"] = df_n[variable].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def generate_specific_time_lags(df:pd.DataFrame, variable:str, n_lags:list):
    """
    Function which lagas
    """
    df_n = df.copy()
    for n in n_lags:
        df_n[f"{variable}_lag{n}"] = df_n[variable].shift(n)
    highest_lag = max(n_lags)
    df_n = df_n.iloc[highest_lag:]
    return df_n



# Reproduction of source code
def time_difference(df, groups, time_idx):
    """
    For debugging time difference assertion erorr.
    There should be no time difference other than 1, when accouting for all in the
    @df: dataframe
    @groups: categorical/group variables
    @time_idx: time step
    """
    g = df.groupby(groups, observed=True)
    df =  -g[time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
    return (df["time_diff_to_next"] != 1).any()



def one_hot_categorical(df, variable):
        """
        """
        onehotencoder = OneHotEncoder()
        onehotencoder.fit(df[[variable]])
    
        encoded = pd.DataFrame(onehotencoder.transform(df[[variable]]).toarray())
        encoded.columns = onehotencoder.categories_[0].tolist()
        cats = onehotencoder.categories_[0].tolist()
        df = pd.concat([df,encoded], axis = 1)
        df.drop(columns = variable, inplace = True)
        df[cats] = df[cats].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")


def scale_features(df, target:str, scaler) -> pd.DataFrame: 
    """
    Scales all features in list columns
    @columns: list or string
    """
    numeric_non_target_columns = df.select_dtypes([int,float]).columns.tolist()
    
    try:
        numeric_non_target_columns.remove(target) 
    except:
        print(" > Target not found")
    for col in numeric_non_target_columns: 
        #print("df col", col)
        df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
    return df


def inverse_scale_features(df, columns, scaler) -> pd.DataFrame:
    """
    Reverse all transformations for input columns in dataframe
    """
    df = df.copy()
    if type(columns) == str:
        columns = [columns]
    for col in columns:
        df[col] = scaler.inverse_transform(df[col].to_numpy().reshape(-1, 1))
    return df


def rolling_mean(series: pd.Series, window = 100) -> pd.Series:
    """
    Exchanges NA's for the rolling average of the operation before and after the observation, 
    constituing the window
    """
    return series.fillna(series.rolling(center=True, window=window, min_periods=1).mean())
    


def remove_nans(df: pd.DataFrame, method: str, order:int = 0) -> pd.DataFrame:
    """
    Removes NAN's using a specified method (assuming linear, splines or polynomial) for
    all columns in dataframe. If one or more of the columns are non-numeric, interpolation method
    "pad" is chosen, i.e, selecting the closest existing value
    """
    df = df.copy()
    for i in range(0, len(df.columns)):
        column = df.columns[i]
        if df.dtypes[i] == np.float64 or df.dtypes[i] == np.int64:
            if order != 0:
                 df[column] = df[column].interpolate(method = method, order =  order)
            else:
                  df[column] = df[column].interpolate(method = method)

        else:    
            df[column] = df[column].interpolate(method = 'pad')
        # if any remaining NA's interpolate using pad
        if any(df[column].isna()):
            df[column] = df[column].fillna(get_first_non_na_row(df[column]))
            
    return df


def remove_nans_cv(df: pd.DataFrame, cv, method: str, order  = 0,  verbose = False) -> pd.DataFrame:
    """
    Removes NA's using a specified method for all splits of an input cross validation type
    """
    df = df.copy()
    for train_indices, val_indices, test_indices in cv.split(df):
        if verbose: 
            print("> train", train_indices)
            print("> val", val_indices)
            print("> test", test_indices)
        df.iloc[train_indices,:] = remove_nans(df.iloc[train_indices,:], method, order)
        df.iloc[val_indices,:]  = remove_nans(df.iloc[val_indices,:], method, order)
        df.iloc[test_indices,:]  = remove_nans(df.iloc[test_indices,:], method, order)
    return df



def get_first_non_na_row(series: pd.Series):
    """
    
    """
    if len(series) == 0:
        raise ValueError("Empty series")
    series = series.dropna()
    if len(series) == 0:
        #raise ValueError("Only NA in series", series)
        return 0
    return series.head(1).values[0]


def all_features_missing_values(df):
    df = pd.DataFrame(df.isnull().sum()/len(df)*100)
    df.columns = ['Missing values (%)']
    df.sort_values(inplace = True, by = 'Missing values (%)', ascending = False)
    df = df.loc[df.iloc[:,0] > 0,:]
    return df

#df = pd.read_pickle("../model_data/tft_testing_series.pkl")


def normalizer(df):
    """
    Normalizing the numerical features using least-squares (l2) from sci-kit learn preprocessor
    """
    df = df.copy()
    select_dtypes_col = df.select_dtypes([int,float]).columns 
    try:
        for col in select_dtypes_col:
            df[col] = normalize(df[col].array.reshape(1,-1)).reshape(-1,1)   
        return pd.DataFrame(df)
    except:
        print("> ERROR: Need to fix missing values first (NaN)")
        print(all_features_missing_values(df))


def variance_detection(df): 
    """
    Returning a data frame with normalized variances
    """
    df = normalizer(df)
    var_df = pd.DataFrame(df.var()).sort_values(by = 0)
    var_df.columns = ['Variance']
    return var_df


def shift_covariates(df: pd.DataFrame, columns: list, h: int) -> pd.DataFrame:
    """ Function that lags a whole data frame postive or negativ periods h"""
    return df[columns].shift(h)

