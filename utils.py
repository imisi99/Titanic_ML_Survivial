import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def modify_data(file_path):
    df = pd.read_csv(file_path)
    print('data before processing...')
    print(df.head())

    df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].fillna('Unknown')

    df['Cabin'] = df['Cabin'].str[0]
    # Convert categorical columns to one-hot encoding
    df = pd.get_dummies(df, columns=['Embarked', 'Parch', 'SibSp', 'Pclass', 'Cabin', 'Sex'], drop_first=True)
    df = df.astype(int)
    print('data after processing...')
    print(df.head())

    return df


def build_model():
    tf.random.set_seed(42)
    model1 = Sequential(
        [
            Dense(120, activation='relu'),
            Dense(60, activation='relu'),
            Dense(30, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')

        ],
        name='model1'
    )

    model2 = Sequential(
        [
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model2'
    )

    model3 = Sequential(
        [
            Dense(330, activation='relu'),
            Dense(165, activation='relu'),
            Dense(82, activation='relu'),
            Dense(41, activation='relu'),
            Dense(20, activation='relu'),
            Dense(10, activation='relu'),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')

        ],
        name='model3'
    )

    return [model1, model2, model3]
