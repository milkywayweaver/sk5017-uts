import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from modules.autolog import ASLT
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.decomposition import PCA
from copy import deepcopy

class OutlierRemover():
    '''
    Removes outlier by using DBSCAN.
    '''
    def __init__(self,**dbscan_kwargs):
        self._pl = Pipeline([
            ('scaler',StandardScaler()),
            ('cluster',DBSCAN(**dbscan_kwargs))
        ])

    def fit_transform(self,df:pd.DataFrame):
        self.df = df
        self.outliers = self._pl.fit_predict(self.df.drop('sex',axis=1))
        self.outliers = np.where(self.outliers == -1,1,0).astype(bool)

        df_dropped = self.df[~self.outliers]
        return df_dropped

    def outlier_pairplot(self):
        self.df['outliers'] = self.outliers
        sns.pairplot(self.df,hue='outliers')
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
            ('categorical',OrdinalEncoder(),self.cols[~self.is_num])
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