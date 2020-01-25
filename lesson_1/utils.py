from sklearn.preprocessing import StandardScaler
import pandas as pd
import category_encoders as ce


def one_hot_encode(cat_data):
    return pd.get_dummies(cat_data, columns=list(cat_data.columns))


def binary_encode(cat_data):
    encoder = ce.BinaryEncoder(cols=list(cat_data.columns))
    df_binary = encoder.fit_transform(cat_data)
    return df_binary


def backward_encode(cat_data):
    encoder = ce.BackwardDifferenceEncoder(cols=list(cat_data.columns))
    df_bd = encoder.fit_transform(cat_data)
    return df_bd


class ScalerTransform:
    def __init__(self, ):
        self.sc = StandardScaler()

    def __call__(self, X, y=None):
        if y is None:
            return self.sc.fit_transform(X)
        return self.sc.fit_transform(X), y


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X, y):
        for trn in self.transforms:
            X, y = trn(X, y)
        return X, y
