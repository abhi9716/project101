!pip install openpyxl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import json
from fbprophet.serialize import model_to_json, model_from_json
import os
model_path = 'ts_model.json'


def preprocess(df, date_col = None, series_col = None):
# date_col = "DATE"
# series_col = "WITHDRAWAL AMT"
    cols = [date_col,series_col]
    df1=df[cols]
    df1 = df1.groupby(date_col).sum()
    # all_days = pd.date_range(df1.index.min(), df1.index.max(), freq='D')
    df1.index = pd.DatetimeIndex(df1.index)
    # df1 = df1.reindex(all_days, fill_value=0)
    df2 = df1.resample("W", label='right').sum()
    df2=df2.reset_index()
    df2.columns = ['ds', 'y']
    df2 = df2[df2.y!=0]
    return df2

def build_model(df, model_path):
    """Define forecasting model."""
    # Create holidays data frame. 
    train = preprocess(df)
    print(len(train))
    model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=False,
#         daily_seasonality=False, 
#         holidays = holidays, 
#         interval_width=0.95, 
#         mcmc_samples = 500
    )

    model.add_seasonality(
        name='monthly', 
        period=30.5, 
        fourier_order=5
    )
    model.fit(train)
    os.remove(model_path)
    with open(model_path, 'w') as fout:
        json.dump(model_to_json(model), fout)  # Save model
    return model


def forcast_next_3week(model):
    future = model.make_future_dataframe(periods=3, freq='W')
    forecast = model.predict(future)
    return forecast[["ds","yhat"]].tail(3).reset_index(drop=True)
forecast = forcast_next_3week(model)

def forcasting(df, date_col = "DATE", series_col = "WITHDRAWAL AMT", model_path=model_path):

    training_data = preprocess(df, date_col = "DATE", series_col = "WITHDRAWAL AMT")
    model = build_model(training_data, model_path)
    forcast_df = forcast_next_3week(model)
    return forcast_df

# 	  ds          	yhat
# 2019-01-13	2.423966e+06
# 2019-01-20	3.246525e+06
# 2019-01-27	2.027275e+06

    


