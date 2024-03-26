import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():
    df = pd.read_csv('../../0_DATASETS/creditcard.csv')
    #print(df.head())
    return df

def data_split_redux(df, target, zero_label_redux=0.5) -> pd.DataFrame:
    """Removes (1 -'zero_label_redux') * rows where target label is 0"""
    idx_to_drop = df[df[target] == 0].sample(frac=zero_label_redux, random_state=42).index
    df_redux = df.drop(labels=idx_to_drop)
    return df_redux

def train_test_valid_split(X, y, train_size=0.6, valid_test_split=0.5) -> tuple:
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, test_size=valid_test_split, stratify=y_, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test