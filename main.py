''' Abigail Chutnik and Kcee Landon
    CSE 163
    Implements functions for
    1. Creating a Machine Learning Model that
    predicts if there is a presence of heart disease
    based on a Chi squared selection of best features.
    Saves a DecisionTree plot and plot of its most imporant
    features
    2. Creating a scatter linear regression plot using a new library
    called Plotly to test with. The plot shows relationships between
    resting blood pressure, age, and sex.
    3. Creating two bar graphs using a new library called Plotly to test
    with. The plots show the correlation coefficients of each heart disease
    attribute with resting angina and exercise-induced angina.
    '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from IPython.display import Image, display
import graphviz


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


def plot_tree(model, features, labels):
    """
    Takes model, features, and labels created for Machine Learning
    Model.
    Creates a DecisionTree of the model and saves it.
    """
    features = list(features.columns)

    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=features,
                                    class_names=True,
                                    impurity=False,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data).render('tree.gv', format='png')
    display(Image(filename='tree.gv.png'))


def q1_best_features(candidates, labels):
    """
    Takes a data frame including potential feature
    column candidates.
    Returns a list of most signficant features based
    on a Chi squared test.
    """
    # selecting the best features using K best features
    # setting up for loop
    selector = SelectKBest(chi2, k=5).fit_transform(candidates, labels)
    first_row = selector[0]
    col_names = list(candidates.columns)
    feature_names = []

    # matching the result of SelectKBest to column names to
    # identify the best features and append to new list
    for f in first_row:
        col = 0
        for val in candidates.iloc[0]:
            if val == f:
                feature_names.append(col_names[col])
            col += 1

    return feature_names


def q1_model(data):
    """
    Takes altered DataFrame.
    Makes and trains a Machine Learning Model
    based on the best features. The model predicts
    if there is a presence of heart disease based on the
    best features.

    Plots a bar graph of important features of the model
    and plots a DecisionTree.
    Saves both plots and returns accuracy scores of the model.
    """
    # making and training a model with best features
    candidates = data.loc[:, data.columns != 'num']
    labels = data['num']
    feature_names = q1_best_features(candidates, labels)
    features = data.loc[:, feature_names]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.33, random_state=4)

    model = DecisionTreeClassifier()
    model = model.fit(features_train, labels_train)

    # plots most significant features of the model
    # plots model's DecisionTree
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.bar(np.array(feature_names)[indices], importances)
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.title("The Importance Scores of the Model's Features")
    plt.savefig("importance.png")
    plot_tree(model, features, labels)

    # predicting the accuracy scores
    train_prediction = model.predict(features_train)
    train_acc = accuracy_score(labels_train, train_prediction)
    test_prediction = model.predict(features_test)
    test_acc = accuracy_score(labels_test, test_prediction)
    print('Accuracy Scores for Training and Testing are:')
    return train_acc, test_acc


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


def q2_plot(data):
    """
    Takes the altered DataFrame.

    Creates a scatter linear regression plot
    of resting blood pressure over patient ages
    and separates them by sex.
    Opens a new web tab with the interactive plot.
    """
    fig = px.scatter(data, x='age', y='trestbps', facet_col='num', color="sex",
                     trendline="ols",
                     title="Resting Blood Pressure over Ages per Sex")
    fig.show()


def perform_data_filtering_q3(data):
    """
    Takes the original DataFrame.
    Returns the altered DataFrame necessary for Q3.
    """
    df = data
    angina_presence = df['cp'] <= 2
    df['cp'] = np.where(angina_presence, 1, 0)

    return df


def q3_plot(data):
    """
    Takes the altered DataFrame.

    Creates a bar graph of calculated correlation
    coefficients of each heart disease attribute with
    resting angina using Plotly.
    Opens a new web tab with the interactive plot.
    Same is done with exercised-induced angina.
    """

    correlations = data.corr(method='pearson').abs()

    # plot for resting angina
    correlations_no_exang = correlations.drop(['exang'])
    cols = ['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'oldpeak', 'slope']
    correlations_no_exang['names'] = cols
    fig = px.bar(correlations_no_exang, x='names', y='cp',
                 title="Correlation: Resting Angina & Attributes")
    fig.show()
    # plot for exercise induced angina
    exercise_corr = correlations.drop(['cp'])
    col_names = ['age', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                 'exang', 'oldpeak', 'slope']
    exercise_corr['names'] = col_names
    new_fig = px.bar(exercise_corr, x='names', y='exang',
                     title="Correlation: Exercise-Induced Angina & Attributes")
    new_fig.show()


def main():
    data = pd.read_csv('cleveland.csv')
    # Q1
    q1_df = perform_data_filtering_q1(data)
    print(q1_model(q1_df))
    # Q2
    q2_df = perform_data_filtering_q2(data)
    q2_plot(q2_df)
    # Q3
    q3_df = perform_data_filtering_q3(data)
    q3_plot(q3_df)


if __name__ == '__main__':
    main()
