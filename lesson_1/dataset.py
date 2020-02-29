import pandas as pd

from utils import one_hot_encode, backward_encode, binary_encode

class Dataset:
    def __init__(self, path, target, cat_preproc_type='one-hot', columns=None , drop=None, transforms=None):
        '''
        :param path: path to dataframe
        :param target: target column name
        :param cat_preproc_type: type of preprocessing for categorical data: 'no-preproc', 'one-hot', 'binary', 'backward'
        :param columns: list of new column names or None, values of target and drop args have to be in this list 
        :param transforms: class Compose or Transform or None
        '''

        self.path = path
        self.data = pd.read_csv(path, index_col=0)
        if 'index' in list(self.data.columns):
            self.data = pd.read_csv(path).drop(columns=['index'])
        if columns is not None:
            self.data.columns = columns
        self.y = self.data[target]
        self.data = self.data.drop(columns=[target])
        if drop is not None:
            self.data = self.data.drop(columns=drop)
        if cat_preproc_type == 'no-preproc':
            self.X = self.data.values
        else:
            self.cat_data = self.data.select_dtypes(include=['object']).copy()

            if cat_preproc_type == 'one-hot':
                self.cat_data = one_hot_encode(self.cat_data)
            elif cat_preproc_type == 'binary':
                self.cat_data = binary_encode(self.cat_data)
            elif cat_preproc_type == 'backward':
                self.cat_data = backward_encode(self.cat_data)
            else:
                raise ValueError("Categorical preprocessing type is not valid.")

            self.X = self.cat_data.join(self.data.select_dtypes(include=['int64', 'float64']))
        
        self.transforms = None
        if transforms is not None:
            if cat_preproc_type == 'no-preproc':
                print('Transforming is impossible when "no-preproc"')
            else:
                self.transforms = transforms

    def get_data(self):
        if self.transforms is not None:
            X, y = self.transforms(self.X, self.y)
        else:
            X, y = self.X, self.y
        return X, y
