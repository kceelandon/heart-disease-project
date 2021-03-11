import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score

sns.set()


def perform_data_filtering_q1(data):
    # redoing the dataframe columns based on different values
    df = data
    df = df.replace('?', np.NaN)
    df = df.dropna()
    df['num'] = df['num'].replace([2, 3, 4], 1)

    return df


def q1_name_later(data):

    # selecting the best features using K best features
    candidates = data.loc[:, data.columns != 'num']
    labels = data['num']

    selector = SelectKBest(chi2, k=5).fit_transform(candidates, labels)
    print(selector[:5])
    first_row = selector[0]
    col_names = list(candidates.columns)
    feature_names = []

    # matching the result of K best features to column names to identify best features
    print('The best features to predict heart disease presence are:')
    for f in first_row:
        col = 0
        for val in candidates.iloc[0]:
            if val == f:
                feature_names.append(col_names[col])
            col += 1
   
    # making a model

    features = data.loc[:, feature_names]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)

    # predicting the accuracy scores

    model = DecisionTreeClassifier()
    model = model.fit(features_train, labels_train)
    
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
    p = sns.relplot(data=data, x='age', y='trestbps', col='num', hue='sex')
    p.set_xlabels('Age')
    p.set_ylabels('Resting Blood Pressure (mmHg)')
    plt.savefig('bpvsage.png', bbox_inches='tight')

def main():
    data = pd.read_csv('cleveland.csv')
    q1_df = perform_data_filtering_q1(data)
    print(q1_name_later(q1_df))
    q2_df = perform_data_filtering_q2(data)
    #q2_plot(q2_df)


if __name__ == '__main__':
    main()