import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def modify_data(file_path, columns_reference=None):
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

    # fitting test data to the columns of the training data
    if columns_reference is not None:
        for col in columns_reference:
            if col not in df.columns:
                df[col] = 0

        df = df[columns_reference]

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


def build_rf():
    rf1 = RandomForestClassifier(n_estimators=400, min_samples_split=2, max_features='sqrt')
    rf2 = RandomForestClassifier(n_estimators=700, min_samples_split=3, max_features='log2')
    rf3 = RandomForestClassifier(n_estimators=1000, min_samples_split=4, max_features='log2')

    rf1.name = 'rf1'
    rf2.name = 'rf2'
    rf3.name = 'rf3'

    return [rf1, rf2, rf3]


def build_xgboost():
    xgb1 = XGBClassifier(n_estimators=400, max_depth=100, learning_rate=0.1)
    xgb2 = XGBClassifier(n_estimators=700, max_depth=300, learning_rate=0.3)
    xgb3 = XGBClassifier(n_estimators=1000, max_depth=200, learning_rate=0.03)

    xgb1.name = 'xgb1'
    xgb2.name = 'xgb2'
    xgb3.name = 'xgb3'

    return [xgb1, xgb2, xgb3]