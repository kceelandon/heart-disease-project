import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

sns.set()

def perform_data_filtering_q2(data):
    # redoing the dataframe columns based on different values
    df = data
    diseased = df['num'] != 0
    df['num'] = np.where(diseased, 'diseased', 'healthy')
    males = df['sex'] == 1
    df['sex'] = np.where(males, 'male', 'female')
    return df

def q2_plot(data):
    sns.relplot(data=data, x='age', y='trestbps', col='num', hue='sex')
    plt.xlabel('Age')
    plt.ylabel('Resting Blood Pressure (mm Hg)')
    plt.title('Blood Pressure vs. Age separated by presence of Disease')
    plt.savefig('bpvsage.png', bbox_inches='tight')

def main():
    data = pd.read_csv('cleveland.csv')
    q2_df = perform_data_filtering_q2(data)
    q2_plot(q2_df)


if __name__ == '__main__':
    main()