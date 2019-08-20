import numpy as np
import warnings

"""
Compute various errors for measuring time series prediction/forecast accuracy

References:
-----------
Books:
Forecasting - Principles and Practice 2e, Hyndman, Athanasopoulos, Section 3.4 (https://otexts.com/fpp2/accuracy.html)
Practical Time Series Forecasting with R - A Hands-On Guide, Shmueli, Lichtendahl, Section 3.3

Papers:
Another look at measures of forecast accuracy, Hyndman, Koehler_2006
Performance Metrics (Error Measures) in Machine Learning Regression, Forecasting and Prognostics - Properties and Typology, Botchkarev, 2018

Wikipedia:
https://en.wikipedia.org/wiki/Root-mean-square_deviation
https://en.wikipedia.org/wiki/Mean_absolute_error
https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
https://en.wikipedia.org/wiki/Average_absolute_deviation
https://en.wikipedia.org/wiki/Mean_squared_error
https://en.wikipedia.org/wiki/Root-mean-square_deviation
https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
"""

# ----------------------------------------------------------------------------------------------------------------------
# Simple Errors
# ----------------------------------------------------------------------------------------------------------------------

def mae_calc(Y, Yhat):
    """
    Mean Absolute Error (MAE)
    See https://otexts.com/fpp2/accuracy.html
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: MAE in the form [num_sequences]
    """
    n_sequences = Y.shape[1]

    mae = []
    for i in range(n_sequences):
        mae.append(np.mean(np.absolute(Y[:, i] - Yhat[:, i])))
    mae = np.array(mae)
    return mae

def mre_calc(Y, Yhat):
    """
    Mean Relative Error (MRE) in percentage
    See BRITS - Bidirectional Recurrent Imputation for Time Series, Cao, Wang, Li, 2018
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: MRE in the form [num_sequences]
    """
    n_sequences = Y.shape[1]

    mre = []
    for i in range(n_sequences):
        mre.append(100 * np.sum(np.absolute(Y[:, i] - Yhat[:, i])) / np.sum(np.absolute(Y[:, i])))
    mre = np.array(mre)
    return mre

# ----------------------------------------------------------------------------------------------------------------------
# Percentage Errors
# ----------------------------------------------------------------------------------------------------------------------

def mape_calc(Y, Yhat):
    """
    Mean Absolute Percentage Error (MAPE)
    See https://otexts.com/fpp2/accuracy.html
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: MAPE in the form [num_sequences]
    """
    n_sequences = Y.shape[1]

    mape = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = Y[:, i] - Yhat[:, i]
        denominator = Y[:, i]
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Calculate error
        mape.append(np.mean(np.absolute(100 * numerator / denominator)))
    mape = np.array(mape)
    return mape

def smape_calc(Y, Yhat):
    """
    # Symmetric Mean Absolute Percentage Error (SAMPE)
    See the M4 Competition documentation and https://otexts.com/fpp2/accuracy.html
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: SMAPE in the form [num_sequences]
    """

    n_sequences = Y.shape[1]

    smape = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = np.absolute(Y[:, i] - Yhat[:, i])
        denominator = (np.absolute(Y[:, i]) + np.absolute(Yhat[:, i]))
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Sequence length
        length = numerator.shape[0]
        # Calculate error
        smape.append(200.0 / length * np.sum(numerator / denominator))
    smape = np.array(smape)
    return smape

# ----------------------------------------------------------------------------------------------------------------------
# Scaled Errors
# ----------------------------------------------------------------------------------------------------------------------

def mase_calc(Y, Yhat):
    """
    Mean Absolute Scaled Error (MASE)
    See https://otexts.com/fpp2/accuracy.html
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: MASE in the form [num_sequences]
    """
    n_sequences = Y.shape[1]
    se = []
    mase = []
    for i in range(n_sequences):
        numerator = (Y[:, i] - Yhat[:, i])
        denominator = np.sum(np.absolute(Y[1:, i] - Y[0:-1, i]), axis=0)
        # Check if denominator is zero
        if denominator == 0:
            warnings.warn("The denominator for the MASE is zero")
            se.append(np.NaN * np.ones(length))
            mase.append(np.NaN)
            continue
        # Sequence length
        length = numerator.shape[0]
        # Scaled Error
        scaled_error = (length - 1) * numerator / denominator
        se.append(scaled_error)
        mase.append(np.mean(np.absolute(scaled_error)))
    mase = np.array(mase)

    return mase

# ----------------------------------------------------------------------------------------------------------------------
# Squared Errors
# ----------------------------------------------------------------------------------------------------------------------

def rmse_calc(Y, Yhat):
    """
    Root Mean Squared Error (RMSE)
    See https://otexts.com/fpp2/accuracy.html
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: RMSE in the form [num_sequences]
    """
    n_sequences = Y.shape[1]

    rmse = []
    for i in range(n_sequences):
        rmse.append(np.mean(np.square(Y[:, i] - Yhat[:, i])))
    rmse = np.array(rmse)
    return rmse

def nrmse_calc(Y, Yhat):
    """
    Normalised Root Mean Squared Error (NRMSE)
    See https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation
    :param Y: Ground truth in the form [sequence_length, num_sequences]
    :param Yhat: Prediction in the form [sequence_length, num_sequences]
    :return: NRMSE in the form [num_sequences]
    """
    # Normalised Root Mean Squared Error
    n_sequences = Y.shape[1]

    nrmse = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = 100 * np.sqrt(np.mean(np.square(Y[:, i] - Yhat[:, i])))
        denominator = np.max(Y[:, i]) - np.min(Y[:, i])
        # Calculate error
        nrmse.append(numerator / denominator)
    nrmse = np.array(nrmse)

    return nrmse

# ----------------------------------------------------------------------------------------------------------------------
# Calculate error functions
# ----------------------------------------------------------------------------------------------------------------------

def calculate_error(Yhat, Y, print_errors=False):
    """
    Calculate various errors on a prediction Yhat given the ground truth Y. Both Yhat and Y can be in the following
    forms:
    * One dimensional arrays
    * Two dimensional arrays with several sequences along the first dimension (dimension 0).
    * Three dimensional arrays with several sequences along first dimension (dimension 0) and with the third dimension
      (dimension 2) being of size 1.
    :param Yhat: Prediction
    :param Y: Ground truth
    """

    # Ensure arrays are 2D
    assert np.ndim(Y) <= 3, 'Y must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Yhat) <= 3, 'Yhat must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Y) <= np.ndim(Yhat), 'Y has a different shape to Yhat'

    # Prepare Y and Yhat based on their number of dimensions
    if np.ndim(Y) == 1:
        n_sequences = 1
        Y = np.expand_dims(Y, axis=1)
        Yhat = np.expand_dims(Yhat, axis=1)
    elif np.ndim(Y) == 2:
        n_sequences = Y.shape[1]
    elif np.ndim(Y) == 3:
        assert Y.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Y = np.squeeze(Y, axis=2)
        assert Yhat.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Yhat = np.squeeze(Yhat, axis=2)
        n_sequences = Y.shape[1]
    else:
        raise Warning('Error in dimensions')

    mae = mae_calc(Y, Yhat)
    if print_errors:
        print('Mean Absolute Error (MAE) = ', mae)

    mre = mre_calc(Y, Yhat)
    if print_errors:
        print('Mean Relative Error (%) (MRE) = ', mre)

    mape = mape_calc(Y, Yhat)
    if print_errors:
        print('Mean Absolute Percentage Error (MAE) = ', mape)
    #
    # smape = smape_calc(Y, Yhat)
    # if print_errors:
    #     print('Symmetric Mean Absolute Percentage Error (sMAPE) = ', smape)
    #
    # mase = mase_calc(Y, Yhat)
    # if print_errors:
    #     print('Mean Absolute Scaled Error (MASE) = ', mase)
    #
    # rmse = rmse_calc(Y, Yhat)
    # if print_errors:
    #     print('Root Mean Squared Error (NRMSE) = ', rmse)
    #
    # nrmse = nrmse_calc(Y, Yhat)
    # if print_errors:
    #     print('Normalised Root Mean Squared Error (NRMSE) = ', nrmse)

    return mae, mre, mape


if __name__ == '__main__':
    x = np.reshape(np.arange(0, 10 * 2), (10, 2)) + np.random.rand(10,2)
    y = x + np.random.rand(10,2)

    # x = np.reshape(np.ones((10, 2)), (10, 2))
    # y = np.copy(x)
    # y[:,1] = y[:,1] + np.random.rand(10)

    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)

    mae, mre, mape = calculate_error(x, y, print_errors=True)
