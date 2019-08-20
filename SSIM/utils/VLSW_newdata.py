import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import random, math, os, time


# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)



def preprocess_df(df):
    """ The training and testing data are manually selected.
    :param df:  dataframe with raw data
    :return:
    """

    # df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.set_index('date', inplace=True)

    print(df.index)

    # Split into train and test
    ## train, test, val
    # df = df.fillna(method='ffill')

    pm25 = df['pm2.5'].values.copy().reshape(-1, 1)

    # Standlization, use StandardScaler
    scaler_x = StandardScaler()
    scaler_x.fit(df[['pm2.5']])
    df[
        ['pm2.5']] = scaler_x.transform(
        df[['pm2.5']])

    scaler_y = StandardScaler()
    scaler_y.fit(pm25)
    y_all = scaler_y.transform(pm25)
    df['pm2.5'] = y_all


    df_train = df.loc['2014/05/01 00:00:00':'2014/2/28 00:00:00'].copy()
    df_test = df.loc['2015/03/01 00:00:00':'2015/04/30 00:00:00'].copy()


    y = y_all.copy()


    return df_train, df_test, y, scaler_x, scaler_y


def generate_samples(x, y, model_params, seq_len_before=7, seq_len_after=7, output_seq_len=9):
    """
    Generate samples, input past and future, target middle
    :param x: input dataframe
    :param y: target variable to impute
    :param seq_len_before:
    :param seq_len_after:
    :param output_seq_len:
    :return: (inputsequence, targetsequence)
    """
    total_samples = x.shape[0]
    total_len = seq_len_before + seq_len_after + output_seq_len

    input_batch_idxs = [list(range(i, i + seq_len_before)) + list(
        range(i + seq_len_before + output_seq_len, i + seq_len_before + output_seq_len + seq_len_after)) for i in
                        range((total_samples - total_len + 1))]

    input_seq = np.take(x, input_batch_idxs, axis=0)

    z = np.zeros((output_seq_len, model_params['dim_in']))

    input_seq = np.array([np.concatenate((i[:seq_len_before], z, i[seq_len_before:])) for i in input_seq])

    output_batch_idxs = [list(range(i + seq_len_before, i + seq_len_before + output_seq_len)) for i in
                         range((total_samples - total_len + 1))]

    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq


def pad_all_cases(x, y, model_params, min_len_before=7, max_len_before=9, min_len_after=7, max_len_after=9,
                  targetlength=9):
    """
    variable length inputs, fix length outputs
    :param x: input dataframe
    :param y: target variable to impute
    :param min_len_before:
    :param max_len_before:
    :param min_len_after:
    :param max_len_after:
    :param targetlength:
    :return: inputsequence with same length, outputsequence with same length
    """
    total_x = []
    total_y = []
    total_len_x = []
    totle_len_before_x = []

    for l_before in range(min_len_before, max_len_before + 1):
        for l_after in range(min_len_after, max_len_after + 1):
            case_x, case_y = generate_samples(x.values, y, model_params, l_before, l_after, targetlength)
            # npad is a tuple of (n_before, n_after) for each dimension


            # print(case_x.shape)
            len_x = np.full(case_x.shape[0],case_x.shape[1])
            len_before_sequence_x = np.full(case_x.shape[0],l_before)
            # print(len_array)

            # npad = ((0, 0), (max_len_before - l_before, max_len_after - l_after), (0, 0))
            npad = ((0, 0), (0, max_len_before - l_before + max_len_after - l_after), (0, 0))

            same_length_x = np.pad(case_x, pad_width=npad, mode='constant', constant_values=0)

            total_x.append(same_length_x)
            total_y.append(case_y)
            total_len_x.append(len_x)
            totle_len_before_x.append(len_before_sequence_x)

    ## total x,y
    concatenated_x = np.concatenate(total_x, axis=0)
    concatenated_y = np.concatenate(total_y, axis=0)
    len_all_case = np.concatenate(total_len_x).ravel()
    len_before_all_case = np.concatenate(totle_len_before_x).ravel()

    return concatenated_x, concatenated_y, len_all_case, len_before_all_case


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    # x_all, y_all, scaler_x, scaler_y = preprocess_df(dataframe)

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(dataframe, dataframe['pm2.5'].values, model_params,
                                                   model_params['min_before'], model_params['max_before'],
                                                   model_params['min_after'], model_params['max_after'],
                                                   model_params['output_length'])

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

    # print(x.shape)
    # print(y.shape)
    # print(x_len.shape)
    # print(x_before_len.shape)

    index_list = []
    ## check and remove samples with NaN (just incase)
    for index, (x_s, y_s, len_s, len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        # print(index)
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    # print('!!!!!!!')
    # print(len(index_list))

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)
    # print(x.shape)
    # print(y.shape)
    # print(x_len.shape)
    # print(x_before_len.shape)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None,
                                                        random_state=SEED,
                                                        shuffle=False)

    x_train_len, x_test_len = train_test_split(x_len, test_size=None, random_state=SEED,shuffle=False)


    x_train_before_len, x_test_before_len = train_test_split(x_before_len, test_size=None, random_state=SEED,shuffle=False)



    return x_train, y_train, x_train_len, x_train_before_len


if __name__ == "__main__":

    sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 15,
        'max_before': 15,
        'min_after': 15,
        'max_after': 15,
        'test_size': 0.2,
        'model_id': 1
    }

    # filepath = r'C:\Users\ZHA244\Coding\Pytorch_based\SSIM\data\simplified_PM25.csv'
    filepath = '../data/pm25_ground.csv'
    df = pd.read_csv(filepath, dayfirst=True, usecols=[0,1],names=["date", "pm2.5"], header=None)
    df.dropna(how="all", inplace=True)
    df = df[1:]

    # print(df.head())
    # print(df.shape)

    df_train, df_test, y, scaler_x, scaler_y = preprocess_df(df)


    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_train, sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len, sampling_params, SEED)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))




    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_test, sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len, sampling_params, SEED)

    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

