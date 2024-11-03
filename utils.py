import pandas as pd


def modify_data(file_path):
    df = pd.read_csv(file_path)
    print('data before processing...')
    print(df.head())

    df = df.drop(columns=['PassengerId', 'Name'])
