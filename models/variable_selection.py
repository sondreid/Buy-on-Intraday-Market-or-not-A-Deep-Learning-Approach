from pandas import read_csv

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utilities import * 

from model_definitions import * 
from matplotlib import pyplot
# load dataset
# separate into input and output variables


series_df = remove_nans(series_df_shifted,  'linear')

series_df_intraday = generate_time_lags(series_df.copy(), 'intraday_price', 50)
series_df_dayahead = generate_time_lags(series_df.copy(), 'dayahead_price', 50)

# Make series with only lags
series_lags_intraday = series_df_intraday[series_df_intraday.columns[pd.Series(series_df_intraday.columns).str.contains("intraday_price_lag")]]
series_lags_intraday['intraday_price'] = series_df['intraday_price']
series_lags_dayahead = series_df_dayahead[series_df_dayahead.columns[pd.Series(series_df_dayahead.columns).str.contains("dayahead_price_lag")]]
series_lags_dayahead['dayahead_price'] = series_df['dayahead_price']





def lag_importance(df:pd.DataFrame, variable:str, n_features : int, n_estimators: int):
	"""
	Performs recursive feature selection

	"""

	print(colored("> Lag selection started", 'green'))
	# perform feature selection
	rfe = RFE(RandomForestRegressor(n_estimators=n_estimators, random_state=123), n_features_to_select=n_features)
	fit = rfe.fit(df[df.columns[~df.columns.isin([variable])]], df[variable])
	# report selected features

	print('> Most important lags:')
	most_important_lags = df.columns.values[0:-1]
	for i in range(len(fit.support_)):
		if fit.support_[i]:
			print(most_important_lags[i])
	# plot feature rank
	most_important_lags = df.columns.values[0:-1]
	ticks = [i for i in range(len(most_important_lags))]
	pyplot.bar(ticks, fit.ranking_)
	pyplot.xticks(ticks, most_important_lags)
	pyplot.show()
	most_important_lags = pd.DataFrame({'most_important_lags':most_important_lags, 'ranking': fit.ranking_})
	if not os.path.exists("lag_selection"):
		os.makedirs("lag_selection")
	most_important_lags.to_pickle("lag_selection/most_important_lags_"+variable + ".pkl")
	print(colored("> Selection complete", 'green'))
	return most_important_lags



most_important_lags_intraday = lag_importance(series_lags_intraday, 'intraday_price', 10, 400)

most_important_lags_dayahead = lag_importance(series_lags_dayahead, 'dayahead_price', 10, 400)



if __name__ == '__main__':

		
	# Plot ACF
	plot_acf(series_df_intraday[['intraday_price']], lags = 100)
	pyplot.show()

	# Plot PACF of intraday
	plot_pacf(series_df_intraday[['intraday_price']], lags = 1500)
	pyplot.show()


	# Plot ACF
	plot_acf(series_df_dayahead[['dayahead_price']])
	pyplot.show()
	weekly_median_intraday = series_df.loc[:,['year','week', 'intraday_price']].groupby(by = ['year','week'], as_index = False).median()
	weekly_median_intraday.year = weekly_median_intraday.year.astype(int)
	weekly_median_intraday.week = weekly_median_intraday.week.astype(int)
	weekly_median_intraday = weekly_median_intraday.sort_values(by = ['year', 'week'], ascending = [True, True])
	weekly_median_intraday['seq']         = np.arange(0, len(weekly_median_intraday)).tolist()



	sns_lineplot(weekly_median_intraday, weekly_median_intraday.week.sort_values(), y = weekly_median_intraday.intraday_price, hue='year')

	# Weekly pacf

	weekly_median_intraday_lags = generate_time_lags(weekly_median_intraday.copy(), 'intraday_price', 60)
	# Plot PACF of intraday
	plot_pacf(weekly_median_intraday[['intraday_price']], lags = 50)
	pyplot.show()
	plot_acf(weekly_median_intraday[['intraday_price']], lags = 50)
	pyplot.show()
