import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from modules.autolog import ASLT
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.decomposition import PCA
from copy import deepcopy

class OutlierRemover():
    '''
    Removes outlier by using LOF.
    '''
    def __init__(self,target=['sex'],n_neighbors=20):
        self._pl = Pipeline([
            ('scaler',StandardScaler()),
            ('outlier',LocalOutlierFactor(novelty=True,n_neighbors=n_neighbors))
        ])
        self.target = target

    def fit(self,df:pd.DataFrame):
        self.is_fitted = True
        self._pl.fit(df.drop(self.target,axis=1))

    def transform(self,df:pd.DataFrame):
        if not hasattr(self,'is_fitted'):
            raise AttributeError('The module has not been fitted. Fit first!')
        self.outliers = self._pl.predict(df.drop(self.target,axis=1))
        self.outliers = np.where(self.outliers == -1,1,0).astype(bool)
        df_dropped = df[~self.outliers]
        return df_dropped,self.outliers

    def fit_transform(self,df:pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def __len__(self):
        if not hasattr(self,'is_fitted'):
            raise AttributeError('The module has not been fitted. Fit first!')
        return self.outliers.sum()

    def outlier_pairplot(self,df:pd.DataFrame,outliers:pd.Series):
        _df = df.copy()
        _df['outliers'] = outliers
        sns.pairplot(_df,hue='outliers')
        plt.show()

class Preprocessor():
    def __init__(self):
        pass

    def fit(self,df:pd.DataFrame):
        self.cols = df.columns
        self.is_num = pd.Series([pd.api.types.is_numeric_dtype(df[col]) for col in df.columns])

        pl = Pipeline([
            ('aslt',ASLT(beta=[-5,-5,30,5,5,5,5,5])),
            ('scaler',StandardScaler()),
            ('pca',PCA())
        ])
        self.ct = ColumnTransformer([
            ('numeric',pl,self.cols[self.is_num]),
            ('categorical',OrdinalEncoder(categories=[['M','F','I']]),self.cols[~self.is_num])
        ],remainder='passthrough')
        self.ct.fit(df)

    def transform(self,df:pd.DataFrame):
        if not hasattr(self,'ct'):
            raise AttributeError('The module has not been fitted. Fit first!')
        transformed_array = self.ct.transform(df)
        return pd.DataFrame(transformed_array,columns=self.cols)

    def fit_transform(self,df:pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def df_until(self,df:pd.DataFrame,step:str):
        if not hasattr(self,'ct'):
            raise AttributeError('The module has not been fitted. Fit first!')
        step_list = self.ct.named_transformers_['numeric'].named_steps
        step_index = list(step_list).index(step)
        steps = list(step_list)[:step_index+1]
        _pl = Pipeline([
            (steps[i],step_list[steps[i]]) for i in range(len(steps))
        ])
        self._ct = deepcopy(self.ct)
        self._ct.set_params(numeric=_pl)

        transformed_array = self._ct.transform(df)
        return pd.DataFrame(transformed_array,columns=self.cols)