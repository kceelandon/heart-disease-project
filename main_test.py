import main
import pandas as pd

DATA = pd.read_csv('cleveland.csv')

def test_filtering_q1():
    assert 2 not in main.perform_data_filtering_q1(DATA)
    assert 3 not in main.perform_data_filtering_q1(DATA)
    assert 4 not in main.perform_data_filtering_q1(DATA)


def test_filtering_q2():
    df = main.perform_data_filtering_q2(DATA)
    sex_column = df['sex'].tolist()
    assert 'male' in sex_column
    assert 'female' in sex_column
    num_column = df['num'].tolist()
    assert 0 not in num_column
    assert 1 not in num_column
    assert 2 not in num_column
    assert 3 not in num_column
    assert 4 not in num_column


def test_filtering_q3():
    df = main.perform_data_filtering_q3(DATA)
    cp_col = df['cp']
    assert 0 in cp_col and 1 in cp_col

def run_tests():
    test_filtering_q1()
    test_filtering_q2()
    test_filtering_q3()

if __name__ == '__main__':
    run_tests()