import boto3
import pandas as pd
from sagemaker import get_execution_role
import sagemaker
import os
import re
import numpy as np
import json
import matplotlib.pyplot as plt
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
import seaborn as sns

def get_test_loss(pred, test):
    test.index = np.arange(len(test))
    test_loss = 0
    for i in range(len(pred)):
        test_loss += np.absolute(pred[0][i] - test['Label'][i])
    test_loss = test_loss/len(pred)
    print('Test loss is {}.'.format(test_loss))

def roi_display(initial_investment, predictions, actual, transaction_rate=0, buy_threshold=0,\
    sell_threshold=0):
    """Returns total return on investment through the following inputs:
    - initial_investment: initial dollar amount to be invested
    - predictions: numpy array of EOD stock price predictions from xgboost model
    - actual: numpy array of actual end of day stock prices
    - transaction_rate: rate to be charged each time transaction is made
    - buy_threshold: prediction percentage of growth before executing buying
        transaction
    - sell_threshold: prediction percentage of loss before executing selling
        transaction
    """

    investing = 0
    holding = initial_investment
    predictions = predictions[0]
    actual = actual['Label']
    actual.index = np.arange(len(predictions))
    
    for i in range(len(predictions)-1):
        if holding > 0:
            if predictions[i+1]*(1-buy_threshold) > predictions[i]:
                investing = (holding*(1-transaction_rate)) / actual[i]
                print('Investing ${:.2f}'.format(holding))
                holding = 0         
        if investing > 0:
            if predictions[i+1]*(1+sell_threshold) < predictions[i]:
                holding = (investing * actual[i])*(1-transaction_rate)
                investing = 0
                print('Holding ${:.2f}'.format(holding))

    if holding > 0:
        print('Total return on investment before taxes is ${:.2f}'.format(\
        holding - initial_investment))
    else:
        print('Total return on investment before taxes is ${:.2f}'.format(\
            investing * actual[(len(actual)-1)] - initial_investment))

def roi(initial_investment, predictions, actual, transaction_rate=0, buy_threshold=0,\
    sell_threshold=0):
    """Returns total return on investment through the following inputs:
    - initial_investment: initial dollar amount to be invested
    - predictions: numpy array of EOD stock price predictions from xgboost model
    - actual: numpy array of actual end of day stock prices
    - transaction_rate: rate to be charged each time transaction is made
    - buy_threshold: prediction percentage of growth before executing buying
        transaction
    - sell_threshold: prediction percentage of loss before executing selling
        transaction
    """

    investing = 0
    holding = initial_investment
    predictions = np.asarray(predictions)
    
    for i in range(len(predictions)-1):
        if holding > 0:
            if predictions[i+1]*(1-buy_threshold) > actual[i]:
                investing = (holding*(1-transaction_rate)) / actual[i]
                holding = 0
        if investing > 0:
            if predictions[i+1]*(1+sell_threshold) < actual[i]:
                holding = (investing * actual[i])*(1-transaction_rate)
                investing = 0

    if holding > 0:
        print('Total return on investment before taxes is ${:.2f}'.format(\
        holding - initial_investment))
    else:
        print('Total return on investment before taxes is ${:.2f}'.format((\
            investing/actual[-1]))*(1-transaction_rate) - initial_investment)


def display_pred_vs_actual(prediction, actual):
    """Return comparison line graphs of predicted and actual stock prices
    - prediciton: array of predicted values of stock prices
    - actual: array of actual stock prices
    """

    assert len(prediction) == len(actual), \
        'Unequal lengths found'

    fig, axes = plt.subplots(ncols=2, figsize=(8,4))
    N=len(prediction)
    ind = np.arange(N)

    ax = axes[0]
    ax.plot(ind, prediction)
    ax.set_title('Prediction')

    ax = axes[1]
    ax.plot(ind, actual)
    ax.set_title('Actual')

    plt.show()

def train_test_split_post_pca(df, ratio, Y_train, Y_test):
    """Split dataframe and return tuples of data and labels to be trained with
        xgboost
    - df: dataframe of compenents after pca
    - ratio: ratio of training vs test data to split !!!Needs to be same as
        ratio provided in train_test_split!!!
    - Y_train, Y_test: already found labels in initial train_test_split
    """

    df_matrix = df.as_matrix()

    train_size = int(df_matrix.shape[0] * ratio)
    X_train = df_matrix[:train_size,:]

    X_test = df_matrix[train_size:,:]

    return (X_train, Y_train), (X_test, Y_test)


def create_transformed_df(train_pca, ltc_scaled_pca, n_top_components):
    """ Return a dataframe of data points with component features.
        - train_pca: A list of pca training data, returned by a PCA model.
        - ltc_scaled_pca: A dataframe of normalized, original features.
        - n_top_components: An integer, the number of top components to use.
        :return: A dataframe, with n_top_component values as columns.
     """
    # create new dataframe to add data to
    ltc_pca_transformed=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        ltc_pca_transformed=ltc_pca_transformed.append([list(components)])

    # keep only the top n components
    start_idx = (ltc_scaled_pca.shape[1]-1) - n_top_components
    ltc_pca_transformed = ltc_pca_transformed.iloc[:,start_idx:]

    # reverse columns, component order
    return ltc_pca_transformed.iloc[:, ::-1]

def display_component(v, features_list, component_num, n_weights=10):
    '''Return barplot displaying how a particular component is made up
    - v: the makeup of the principal components
    - features_list: column names of the initial dataframe
    - component_num: component to display, numbered from most variance to least
    - n_weights: number of weights to display in barplot
    '''
    # get index of component (last row - component_num)
    row_idx = (len(v)-1)-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)),
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(\
        n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data,
                   x="weights",
                   y="features",
                   palette="Greens_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()

def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
    Returns the expected data variance covered by the n_top_components.
    -s: A dataframe of singular values for top components;
        the top value is in the last row.
    -n_top_components: An integer, the number of top components to use.
    '''

    start_idx = len(s) - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()

    return exp_variance[0]


def train_test_split(df, ratio):
    """Split dataframe and return tuples of data and labels to be trained with
        xgboost
    - df: dataframe to of data with label in last column
    - ratio: ratio of training vs test data to split
    """

    df_matrix = df.as_matrix()

    train_size = int(df_matrix.shape[0] * ratio)
    X_train = df_matrix[:train_size, : -1]
    Y_train = df_matrix[:train_size, -1]

    X_test = df_matrix[train_size:, : -1]
    Y_test = df_matrix[train_size:, -1]

    return (X_train, Y_train), (X_test, Y_test)
