import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from datetime import datetime
import random

filepath = r'C:\Users\ZHA244\Coding\Production_App\Bi_Attention\DARNN\univariate\data\beijing\PRSA_data_2010.1.1-2014.12.31.csv'
df = pd.read_csv(filepath)

df['date'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: f"{row['year']}-{row['month']}-{row['day']} {row['hour']}:00:00", axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df.drop(['No','year','month','day','hour'], axis=1, inplace=True)

df.sort_values(by='date', inplace=True)

df.set_index('date', inplace=True)

# df['pm2.5'].fillna(method='ffill', inplace=True)

le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])

one_hot = pd.get_dummies(df['cbwd'], prefix='winddirect')
# Drop column B as it is now encoded
df = df.drop('cbwd',axis = 1)
# Join the encoded df
df = df.join(one_hot)


# # add feature
# df['Hourofday'] = df.index.hour
# df['Dayofweek'] = df.index.dayofweek
# df['Month'] = df.index.month
#
# # One-hot encode 'Hourofday'
# temp = pd.get_dummies(df['Hourofday'], prefix='Hourofday')
# df = pd.concat([df, temp], axis=1)
# del df['Hourofday'], temp
#
#
# # One-hot encode 'Dayofweek'
# temp = pd.get_dummies(df['Dayofweek'], prefix='Dayofweek')
# df = pd.concat([df, temp], axis=1)
# del df['Dayofweek'], temp
#
#
# # One-hot encode 'Month'
# temp = pd.get_dummies(df['Month'], prefix='Month')
# df = pd.concat([df, temp], axis=1)
# del df['Month'], temp


PM25_new = df.to_csv('./newPM.csv')


# print(df.head())