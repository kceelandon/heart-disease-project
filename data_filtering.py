import numpy as np


def perform_data_filtering_q1(data):
    """
    Takes the original DataFrame.
    Returns the altered DataFrame necessary for Q1.
    """
    df = data
    df = df.replace('?', np.NaN)
    df = df.dropna()
    df['num'] = df['num'].replace([2, 3, 4], 1)

    return df


def perform_data_filtering_q2(data):
    """
    Takes the original DataFrame.
    Returns the altered DataFrame necessary for Q2.
    """
    # redoing the dataframe columns based on different values
    df = data
    diseased = df['num'] != 0
    df['num'] = np.where(diseased, 'diseased', 'healthy')
    males = df['sex'] == 1
    df['sex'] = np.where(males, 'male', 'female')
    return df


def perform_data_filtering_q3(data):
    """
    Takes the original DataFrame.
    Returns the altered DataFrame necessary for Q3.
    """
    df = data
    angina_presence = df['cp'] <= 2
    df['cp'] = np.where(angina_presence, 1, 0)

    return df
