import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import random, math, os, time

from utils.VLSW import pad_all_cases
# from VLSW import pad_all_cases

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


def preprocess_df(df):
    """ The training and testing data are manually selected.
    :param df:  dataframe with raw data
    :return:
    """

    df.set_index('Timestamp', inplace=True)

    ## some variables are not used in training the model, based on the performance evaluation
    df.drop(['Dayofweek'], axis=1, inplace=True)
    df.drop(['Month'], axis=1, inplace=True)

    tw = df['NO3'].values.copy().reshape(-1, 1)

    # Standlization, use StandardScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(
        df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']])
    df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']] = scaler_x.transform(df[[
            'Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level'
        ]])

    scaler_y = MinMaxScaler()
    scaler_y.fit(tw)
    y_all = scaler_y.transform(tw)

    # get data from 2014 and 2015
    # 6，7, 8, 9，10 as train; 11 as test

    # df_train_one = df.loc['2019-04-01T00:00':'2019-12-31T23:00'].copy()
    # # df_train_two = df.loc['2015-06-01T00:00':'2015-10-31T23:30'].copy()

    # df_test_one = df.loc['2019-01-01T00:00':'2019-03-31T23:00'].copy()
    # # df_test_two = df.loc['2015-11-01T00:00':'2015-11-30T23:30'].copy()

    ##### for dual-head comparision #########
    df_train_one = df.loc['2019-01-01T00:00':'2019-09-30T23:00'].copy()
    df_test_one = df.loc['2019-10-01T00:00':'2019-12-31T23:00'].copy()


    # return df_train_one, df_train_two, df_test_one, df_test_two, scaler_x, scaler_y

    return df_train_one, df_test_one, scaler_x, scaler_y


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(dataframe,
                                                                                            dataframe['NO3'].values,
                                                                                            model_params,
                                                                                            model_params['min_before'],
                                                                                            model_params['max_before'],
                                                                                            model_params['min_after'],
                                                                                            model_params['max_after'],
                                                                                            model_params[
                                                                                                'output_length'])

    train_val_test_y = np.expand_dims(train_val_test_y, axis=2)

    return train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples


def train_test_split_SSIM(x, y, x_len, x_before_len, model_params, SEED):
    '''
    :param x: all x samples
    :param y: all y samples
    :param model_params: parameters
    :param SEED: random SEED
    :return: train set, test set
    '''

    ## check and remove samples with NaN (just incase)
    index_list = []
    for index, (x_s, y_s, len_s, len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None,
    #                                                     random_state=SEED,
    #                                                     shuffle=False)

    # x_train_len, x_test_len = train_test_split(x_len, test_size=None, random_state=SEED, shuffle=False)

    # x_train_before_len, x_test_before_len = train_test_split(x_before_len, test_size=None, random_state=SEED,
    #                                                          shuffle=False)

    # return x_train, y_train, x_train_len, x_train_before_len
    return x, y, x_len, x_before_len


def test_qld_single_station():
    train_sampling_params = {
        'dim_in': 6,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    test_sampling_params = {
        'dim_in': 6,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    filepath = 'SSIM/data/QLD_nomiss.csv'

    
    df = pd.read_csv(filepath)

    df_train_one, df_test_one, scaler_x, scaler_y = preprocess_df(df)

    print('train_preprocess:{}'.format(df_train_one.shape))
    print('test_preprocess:{}'.format(df_test_one.shape))


    # train one
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train_one, train_sampling_params)

    x_train_one, y_train_one, x_train_len_one, x_train_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)


    x_train = x_train_one
    y_train = y_train_one

    x_train_len = x_train_len_one
    x_train_before_len = x_train_before_len_one

    # test one

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_one, test_sampling_params)

    x_test_one, y_test_one, x_test_len_one, x_test_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # # test two

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_test_two, test_sampling_params)

    # x_test_two, y_test_two, x_test_len_two, x_test_before_len_two = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # # concate all test data

    # x_test = np.concatenate((x_test_one, x_test_two), axis=0)
    # y_test = np.concatenate((y_test_one, y_test_two), axis=0)

    x_test = x_test_one
    y_test = y_test_one

    x_test_len = x_test_len_one
    x_test_before_len = x_test_before_len_one


    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # x_train = x_train[:4930]
    # y_train = y_train[:4930]
    # x_test = x_test[:1600]
    # y_test = y_test[:1600]
    # x_train_len = x_train_len[:4930]
    # x_train_before_len = x_train_before_len[:4930]
    # x_test_len = x_test_len[:1600]
    # x_test_before_len = x_test_before_len[:1600]




    return (x_train, y_train, x_train_len, x_train_before_len) , (x_test, y_test, x_test_len, x_test_before_len), (scaler_x, scaler_y)



if __name__ == "__main__":
    # train_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 5,
    #     'min_before': 20,
    #     'max_before': 25,
    #     'min_after': 20,
    #     'max_after': 25,
    #     'file_path': '../data/simplified_PM25.csv'
    # }

    # test_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 5,
    #     'min_before': 25,
    #     'max_before': 25,
    #     'min_after': 25,
    #     'max_after': 25,
    #     'file_path': '../data/simplified_PM25.csv'
    # }

    # filepath = '../data/simplified_PM25.csv'
    # df = pd.read_csv(filepath, dayfirst=True)

    # df_train, df_test, y, scaler_x, scaler_y = preprocess_df(df)

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_train, train_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len,
    #                                                                           train_sampling_params, SEED)

    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_test, test_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len,
    #                                                                       test_sampling_params, SEED)

    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))

    test_qld_single_station()