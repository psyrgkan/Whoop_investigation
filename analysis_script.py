# %%
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import datetime as dt

# %% [markdown]
# ----
# ### Import CSVs from Whoop

# %%
physio = pd.read_csv(r'/Users/tinym/Desktop/my_whoop_data_2022_11_28/physiological_cycles.csv')

# %%
physio.columns

# %%
physio.head()

# %%
physio.shape

# %%
sleep = pd.read_csv(r'/Users/tinym/Desktop/my_whoop_data_2022_11_28/sleeps.csv')

# %%
sleep.shape

# %%
sleep.head()

# %%
workout = pd.read_csv(r'/Users/tinym/Desktop/my_whoop_data_2022_11_28/workouts.csv')

# %%
workout.shape

# %%
workout.dtypes

# %%
workout['Energy burned (cal)'].hist(bins=10)

# %%
workout.loc[workout['Activity name'] == 'Surfing'].head()

# %%
workout.head(10)

# %% [markdown]
# ----
# ### Import XMLs from Apple Health

# %%
# create element tree object 
tree = ET.parse(r'/Users/tinym/Desktop/apple_health_export/export.xml') 

# for every health record, extract the attributes into a dictionary (columns). Then create a list (rows).
root = tree.getroot()

# %%
record_list = [x.attrib for x in root.iter('Record')]

# create DataFrame from a list (rows) of dictionaries (columns)
data = pd.DataFrame(record_list)

# proper type to dates
for col in ['creationDate', 'startDate', 'endDate']:
    data[col] = pd.to_datetime(data[col])

# value is numeric, NaN if fails
data['value'] = pd.to_numeric(data['value'], errors='coerce')


# %%
# some records do not measure anything, just count occurences
# filling with 1.0 (= one time) makes it easier to aggregate
data['value'] = data['value'].fillna(1.0)

# shorter observation names
data['type'] = data['type'].str.replace('HKQuantityTypeIdentifier', '')
data['type'] = data['type'].str.replace('HKCategoryTypeIdentifier', '')
data.head()


# %%
data.type.unique()

# %%
hr = data.loc[data.type == 'HeartRate']

# %%
hr.shape

# %%
hr.dtypes

# %%
hr.sort_values(by='value', ascending=False).tail(50)

# %%
start = dt.datetime(2022,8,20,18,7,59)
finish = dt.datetime(2022,8,20,19,47,59)

# %%
start = dt.datetime(2021,10,7, 9,25,26)
finish = dt.datetime(2021,10,7, 10,2,26)

# %%
hr.startDate = hr.startDate.dt.tz_localize(None)

# %%
hr.loc[(hr.startDate >= np.datetime64(start)) & (hr.startDate <= np.datetime64(finish))]

# %%
hr.loc[(hr.startDate >= np.datetime64(start)) & (hr.startDate <= np.datetime64(finish))].value.plot()

# %%
steps = data.loc[data.type == 'StepCount']

# %%
steps.head()

# %%
data.sourceName.unique()

# %% [markdown]
# *** 
# ### Processing Steps 

# %%
steps.creationDate = steps.creationDate.dt.date

# %%
steps.head()

# %%
stepsByDate = steps[['creationDate', 'value']].groupby('creationDate').sum()

# %%
stepsByDate.head()

# %%
type(stepsByDate)

# %%
stepsByDate.plot();

# %% [markdown]
# ***
# ### Processing physio

# %%
physio["Date"] = pd.to_datetime(physio['Wake onset']).dt.date

# %%
physio.head()

# %%
physio.dtypes

# %% [markdown]
# *** 
# ### Processing workouts

# %%
workout["Date"] = pd.to_datetime(workout['Cycle start time']).dt.date

# %%
workout.head()

# %%
workout.drop(['Cycle start time', 'Cycle end time', 'Cycle timezone', 'Workout start time', 'Workout end time', 
                'Distance (meters)', 'Altitude gain (meters)', 'Altitude change (meters)', 'GPS enabled'], inplace=True, axis=1)

# %%
workout.dtypes

# %%
workout.head()

# %% [markdown]
# ***
# ### Processing Sleep

# %%
sleep["Date"] = pd.to_datetime(sleep['Cycle start time']).dt.date

# %%
sleep.head()

# %%
sleep['Sleep onset'] = pd.to_datetime(sleep['Sleep onset']) 
sleep['Wake onset'] = pd.to_datetime(sleep['Wake onset']) 

# %%
sleep['Hour sleep'] = sleep['Sleep onset'].dt.hour
sleep['Hour wake'] = sleep['Wake onset'].dt.hour

# %%
sleep.drop(['Cycle start time', 'Cycle end time', 'Cycle timezone', 'Sleep onset', 'Wake onset'], inplace=True, axis=1)

# %%
sleep.dtypes

# %%
sleep.head()

# %% [markdown]
# ***
# ### PyMySQL & SQLalchemy

# %%
sleep.head()

# %%
sqlphysio = physio

# %%
for col in sleep.columns:
    try:
        if col != 'Date':
            sqlphysio = sqlphysio.drop(col, axis=1)
    except:
        print(col, "col not in physio")

# %%
sqlphysio.drop(['Sleep onset', 'Wake onset'], axis=1, inplace=True)

# %%
sqlphysio.drop(['Cycle start time', 'Cycle end time', 'Cycle timezone'], axis=1, inplace=True)

# %%
sqlphysio.head()

# %%
workout.head()

# %%
stepsByDate.reset_index(inplace=True)

# %%
stepsByDate.head()

# %%
sqlphysio.head()

# %%
from sqlalchemy import create_engine

user = 'root'
host = '127.0.0.1'
password = 'Syrgkbas741954'
port = 3306
database = 'whoop'

my_conn = create_engine("mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        ))
        
db_conn = my_conn.connect()

# %%
try:
    frame = workout.to_sql("workout", db_conn, if_exists='fail')
    frame = sleep.to_sql("sleep", db_conn, if_exists='fail')
    frame = sqlphysio.to_sql("physio", db_conn, if_exists='fail')
    frame = stepsByDate.to_sql("steps", db_conn, if_exists='fail')

except ValueError as vx:
    print(vx)

except Exception as ex:   
    print(ex)

else:
    print("Table %s created successfully."%"df");   

finally:
    db_conn.close()

# %% [markdown]
# ***
# ### Joining
# Fixing dates and joining on them

# %%
df = physio.merge(stepsByDate, left_on='Date', right_on='creationDate', how='left')

# %%
df.drop('creationDate', axis=1, inplace=True)

# %%
df.head()

# %%
df.rename(columns = {'value':'Steps'}, inplace = True)

# %%
df.head()

# %%
df.drop(['Cycle start time', 'Cycle end time'], inplace=True, axis=1)

# %%
df.columns

# %%
df = df[['Date', 'Resting heart rate (bpm)',
       'Heart rate variability (ms)', 'Skin temp (celsius)', 'Blood oxygen %',
       'Energy burned (cal)', 'Max HR (bpm)', 'Average HR (bpm)',
       'Sleep onset', 'Wake onset', 'Sleep performance %',
       'Respiratory rate (rpm)', 'Asleep duration (min)',
       'Light sleep duration (min)', 'Deep (SWS) duration (min)',
       'REM duration (min)', 'Awake duration (min)', 'Sleep need (min)',
       'Sleep debt (min)', 'Steps', 'Day Strain', 'Recovery score %']]

# %%
df.head()

# %% [markdown]
# ***
# ### Correlations

# %%
del steps
del root
del record_list
del hr
del data
del tree

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
def plot_cor(data):
	corr = data.corr()
	mask = np.triu(np.ones_like(corr))
	ax = sns.heatmap(
		corr, mask=mask,
		vmin=-1, vmax=1, center=0,
		cmap=sns.diverging_palette(20, 220, n=200),
		square=True
	)
	ax.set_xticklabels(
		ax.get_xticklabels(),
		rotation=45,
		horizontalalignment='right'
	);

# %%
plot_cor(df)

# %%
def drop_cor(df, trsh=0.9):
	# Create correlation matrix
	corr_matrix = df.corr().abs()
	
	# Select upper triangle of correlation matrix
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
								k=1).astype(bool))
	
	# Find features with correlation greater than 0.9
	to_drop = [column for column in upper.columns if any(upper[column] > trsh)]
	
	# Drop features 
	df.drop(to_drop, axis=1, inplace=True)

# %%
# drop_cor(df)

# %%
plot_cor(df)

# %% [markdown]
# ***
# ### Date Handling

# %%
dates = df["Date"].tolist()
dates

# %%
start_date = dates[-2]
end_date = dates[0]

# Create a boolean mask that identifies the rows in the dataframe that fall within the specified date range
mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)

# Use the mask to filter the dataframe and select only the rows that fall within the date range
df_filtered = df[mask]

# Print the unique dates in the filtered dataframe
# print(df_filtered["Date"].unique())


# %%
import datetime

# %%
date_range = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days)]

# Subtract the list of unique dates from the full range of dates to find the missing dates
missing_dates = [date for date in date_range if date not in df_filtered["Date"].unique()]

# Print the missing dates
print(missing_dates)


# %% [markdown]
# This looks to be the most recent date when I started using my WHOOP again

# %%
start_date = datetime.date(2021, 9, 20)
end_date = datetime.date(2022, 11, 27)

# %%
date_range = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days)]

# Subtract the list of unique dates from the full range of dates to find the missing dates
missing_dates = [date for date in date_range if date not in df_filtered["Date"].unique()]

# Print the missing dates
print(len(missing_dates))


# %%
# Create a boolean mask that identifies the rows in the dataframe that fall within the specified date range
mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)

# Use the mask to filter the dataframe and select only the rows that fall within the date range
df_recent = df[mask]

# Print the unique dates in the filtered dataframe
df_recent.head()

# %%
df_recent.tail()

# %% [markdown]
# ***
# ### NaN Handling

# %%
df_recent.columns

# %%
df_recent.describe()

# %%
blood_mode = df_recent['Blood oxygen %'].mode()[0]

# %%
skin_mode = df_recent['Skin temp (celsius)'].mode()[0]

# %%
df_recent['Blood oxygen %'].replace(np.nan, blood_mode, inplace=True)

# %%
df_recent['Skin temp (celsius)'].replace(np.nan, skin_mode, inplace=True)

# %%
df_recent.isna().sum()

# %%
df_recent.describe()

# %%
df_recent["next_recovery"] = df_recent["Recovery score %"].shift(-1)


# %% [markdown]
# ***
# ### Time Handling in physio

# %% [markdown]
# Here I wanted to turn the time of waking up and falling asleep to simple hours so that it would potentially lead to some better insights

# %%
df_recent['Sleep onset'] = pd.to_datetime(df_recent['Sleep onset']) 
df_recent['Wake onset'] = pd.to_datetime(df_recent['Wake onset']) 

# %%
df_recent.dtypes

# %%
df_recent['Hour sleep'] = df_recent['Sleep onset'].dt.hour
df_recent['Hour wake'] = df_recent['Wake onset'].dt.hour

# %%
df_recent.head()

# %% [markdown]
# ***
# ### Pandas Profiling

# %%
# from pandas_profiling import ProfileReport
# profile = ProfileReport(df_recent, title='My WHOOP Data', explorative = True)
# profile

# %% [markdown]
# *** 
# ### EDA of physio

# %%
plot_cor(df_recent);

# %%
sns.histplot(df_recent['Resting heart rate (bpm)'], kde=True);

# %%
sns.histplot(df_recent['Heart rate variability (ms)'], kde=True);

# %%
sns.histplot(df_recent['Hour wake'], kde=True);

# %%
sns.histplot(df_recent['Hour sleep'], kde=True);

# %%
df_recent['Adj Hour sleep'] = (df_recent['Hour sleep'] - 6)%12

# %%
sns.histplot(df_recent['Adj Hour sleep'], kde=True);

# %%
sns.histplot(df_recent['Day Strain'], kde=True);

# %%
sns.histplot(df_recent['Recovery score %'], kde=True);

# %%
sns.regplot(data=df_recent, y='Recovery score %', x='Resting heart rate (bpm)')

# %%
sns.regplot(data=df_recent, y='Recovery score %', x='Heart rate variability (ms)')

# %%
sns.regplot(data=df_recent, y='Day Strain', x='Average HR (bpm)')

# %%
df_recent.columns

# %%
sns.regplot(data=df_recent, y='Day Strain', x='Steps')

# %% [markdown]
# *** 
# ### Time series analysis of RHR and HRV

# %%
df_sorted = df_recent.sort_values(by='Date').reset_index()

# %%
df_sorted.drop('index', axis=1, inplace=True)

# %%
df_sorted["next_recovery"] = df_sorted["Recovery score %"].shift(-1)

# %%
df_sorted.head()

# %%
df_sorted['SMA30'] = df_sorted['Resting heart rate (bpm)'].expanding(30).mean()

df_sorted[['Resting heart rate (bpm)', 'SMA30']].plot()


# %%
df_sorted['SMA30HRV'] = df_sorted['Heart rate variability (ms)'].expanding(30).mean()

df_sorted[['Heart rate variability (ms)', 'SMA30HRV']].plot()


# %%
df_sorted.columns

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tools.eval_measures import rmse

# %%
result1 = adfuller(df_sorted['Resting heart rate (bpm)'], autolag='AIC')
print(f'ADF Statistic: {result1[0]}')
print(f'p-value: {result1[1]}')

# %%
result2 = kpss(df_sorted['Resting heart rate (bpm)'])
print(f'KPSS Statistic: {result2[0]}')
print(f'p-value: {result2[1]}')

# %%
result1 = adfuller(df_sorted['Heart rate variability (ms)'], autolag='AIC')
print(f'ADF Statistic: {result1[0]}')
print(f'p-value: {result1[1]}')

# %%
result2 = kpss(df_recent['Heart rate variability (ms)'])
print(f'KPSS Statistic: {result2[0]}')
print(f'p-value: {result2[1]}')

# %%
plot_acf(df_sorted['Resting heart rate (bpm)'], lags=30)

# Show the data as a plot (via matplotlib)
plt.show();

# %%
plot_acf(df_recent['Heart rate variability (ms)'], lags=30)

# Show the data as a plot (via matplotlib)
plt.show();

# %%
plot_pacf(df_sorted['Resting heart rate (bpm)'], alpha =0.05, lags=20)

plt.show();

# %%
plot_pacf(df_recent['Heart rate variability (ms)'], alpha =0.05, lags=20)

plt.show();

# %%
decomp_m = seasonal_decompose(df_sorted['Resting heart rate (bpm)'], model='multiplicative', period=27)

decomp_m.plot()
plt.show()

# %%
decomp_m = seasonal_decompose(df_sorted['Heart rate variability (ms)'], model='multiplicative', period=27)

decomp_m.plot()
plt.show()

# %%
stepwise_fit = pm.auto_arima(df_sorted['Resting heart rate (bpm)'], start_p=0, start_q=0,
                             m=15, max_p=30, max_q=30,
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True, scoring='mse')  # set to stepwise


# %%
opt = ARIMA(df_sorted['Resting heart rate (bpm)'], order=(1,1,2), seasonal_order=(0,0,0,15))
rm_opt = opt.fit()
print(rm_opt.summary())

# %%
print(rmse(rm_opt.predict(), df_sorted['Resting heart rate (bpm)']))

# %%
df_sorted['Resting heart rate (bpm)'].describe()

# %%
stepwise_fit = pm.auto_arima(df_sorted['Heart rate variability (ms)'], start_p=0, start_q=0,
                             m=15, max_p=30, max_q=30,
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True, scoring='mse')  # set to stepwise


# %%
opthrv = ARIMA(df_sorted['Heart rate variability (ms)'], order=(0,1,1), seasonal_order=(0,0,0,15))
rm_opt_hrv = opthrv.fit()
print(rm_opt_hrv.summary())

# %%
print(rmse(rm_opt_hrv.predict(), df_sorted['Heart rate variability (ms)']))

# %%
df_sorted['Heart rate variability (ms)'].describe()

# %%
rm_opt.forecast(steps=20)

# %%
rm_opt_hrv.forecast(steps=20)

# %% [markdown]
# ***
# ### Recovery Regression

# %%
# from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# %% [markdown]
# #### Next day recovery prediction

# %%
df_sorted["next_sleep"] = df_sorted['Asleep duration (min)'].shift(-1)
df_sorted = df_sorted[df_sorted['next_recovery'].notna()]
df_sorted = df_sorted[df_sorted['Asleep duration (min)'].notna()]
X = df_sorted.drop(['next_recovery', 'Sleep onset', 'Wake onset', 'Date'], axis = 1)
y = df_sorted['next_recovery']

# %%
X.fillna(method='backfill', inplace=True)

# %%
X.isna().sum()

# %%
X.dtypes

# %%
SFM = SelectFromModel(estimator=RandomForestClassifier())
sel = SFM.fit(X, y)
selected_feat= X.columns[(sel.get_support())]
print(selected_feat)

# %%
X = X.loc[: , selected_feat]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

# %%
y.head()

# %%
y.plot()

# %%
from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_ml_pipeline.py')

# %%
from sklearn.linear_model import LassoLarsCV, ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import MinMaxScaler


# %%
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, results))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, results))

# %%
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# %%
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# %%
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive


# %%
training_features, testing_features, training_target, testing_target = \
            train_test_split(X, y, random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.01)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.01, fit_intercept=False, l1_ratio=0.25, learning_rate="invscaling", loss="squared_error", penalty="elasticnet", power_t=100.0)),
    RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=16, min_samples_split=14, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# %% [markdown]
# #### Current day recovery

# %%
X = df_sorted.drop(['Recovery score %', 'next_sleep', 'next_recovery', 'Sleep onset', 'Wake onset', 'Date'], axis = 1)
y = df_sorted['Recovery score %']

# %%
X.fillna(method='backfill', inplace=True)

# %%
X.isna().sum()

# %%
X.dtypes

# %%
SFM = SelectFromModel(estimator=RandomForestClassifier())
sel = SFM.fit(X, y)
selected_feat= X.columns[(sel.get_support())]
print(selected_feat)

# %%
X = X.loc[: , selected_feat]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

# %%
y.head()

# %%
y.plot()

# %%
from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_ml_pipeline1.py')

# %%
from sklearn.linear_model import LassoLarsCV, ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# %%
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV
from tpot.builtins import StackingEstimator

# %%
exported_pipeline = XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=14, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9500000000000001, verbosity=0)
exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, results))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, results))


# %%
reg = LassoLarsCV()
regi = reg.fit(X_train, y_train) 
y_pred = regi.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# %%



