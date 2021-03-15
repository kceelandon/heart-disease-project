import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from IPython.display import Image, display
import graphviz
from sklearn.tree import export_graphviz

sns.set()


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
    Takes model, features, and labels created for ML.
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

    # matching the result of SelectKBest to column names to identify best features
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
    based on the best features.

    Plots a bar graph of important features and
    plots DecisionTree.
    Returns both plots and the accuracy scores of the model.
    """
    # making and training a model
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

    return train_acc, test_acc


def perform_data_filtering_q2(data):
    # redoing the dataframe columns based on different values
    df = data
    diseased = df['num'] != 0
    df['num'] = np.where(diseased, 'diseased', 'healthy')
    males = df['sex'] == 1
    df['sex'] = np.where(males, 'male', 'female')
    return df


def q2_plot(data):

    fig = px.scatter(data, x='age', y='trestbps', facet_col='num', color="sex", trendline="ols")
    #fig.show()


def perform_data_filtering_q3(data):
    df = data
    angina_presence = df['cp'] <= 2
    df['cp'] = np.where(angina_presence, 1, 0)
    
    return df


def q3(data):
    ''' what do we want to do here? getting angina presence
        based on factors is a classifier problem but
        it sounds like we wanna do linear regression
        using plotly?'''

    correlations = data.corr().abs()
    # for resting angina
    fig = px.bar(correlations, x=, y='cp')
    fig.show
    # for exercise induced angina
    exercise_corr = correlations.loc[['exang']].abs()
    print(rest_corr)
    print(exercise_corr)


def main():
    data = pd.read_csv('cleveland.csv')
    q1_df = perform_data_filtering_q1(data)
    q1_model(q1_df)
    q2_df = perform_data_filtering_q2(data)
    q2_plot(q2_df)
    q3_df = perform_data_filtering_q3(data)
    print(q3(q3_df))


if __name__ == '__main__':
    main()