import pandas as pd


def modify_data(file_path):
    df = pd.read_csv(file_path)
    print('data before processing...')
    print(df.head())

    df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].fillna('Unknown')

    df['Cabin'] = df['Cabin'].str[0]
    # Convert categorical columns to one-hot encoding
    df = pd.get_dummies(df, columns=['Embarked', 'Parch', 'SibSp', 'Pclass', 'Cabin', 'Sex'], drop_first=True)
    df = df.astype(int)
    print('data after processing...')
    print(df.head())

    return df
