import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random, math, os, time

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


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

            len_x = np.full(case_x.shape[0], case_x.shape[1])
            len_before_sequence_x = np.full(case_x.shape[0], l_before)

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

