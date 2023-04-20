


import pandas as pd
import numpy as np
import os
import plotly.express as px
import re

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (data frame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than this will be removed
      upper (numeric): observations with values higher than this will be removed
    
    Output:
      a winsorized data frame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    """  
    return data[(data[variable]>lower) & (data[variable]<upper)]


def Log_Trans(data,col):
    data["Log "+col] = np.log(data[col])
    return data


# +
def extract_Room(description):
    match = re.search(r'(\d+)(?=\s+of which are bedrooms)',description)
    if match:
        return int(match.group(1))
    else:
        return 0

def add_total_bedrooms(data):
    """
    Input:
      data (data frame): a data frame containing at least the Description column.
    """
    with_rooms = data.copy()
    with_rooms['Bedrooms'] = with_rooms['Description'].apply(extract_Room)
    return with_rooms


# -

def find_expensive_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (data frame): should contain at least a string-valued Neighborhood
        and a numeric 'Sale Price' column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices
    
    Output:
      a list of the top n richest neighborhoods as measured by the metric function
    """
    neighborhoods = training_data.groupby('Neighborhood Code')['Log Sale Price'].agg(metric).sort_values(ascending=False).head(n).index
    
    # This makes sure the final list contains the generic int type used in Python3, not specific ones used in numpy.
    return [int(code) for code in neighborhoods]
def add_in_expensive_neighborhood(data, neighborhoods):
    """
    Input:
      data (data frame): a data frame containing a 'Neighborhood Code' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as rich
    Output:
      data frame identical to the input with the addition of a binary
      in_rich_neighborhood column
    """
    
    data['in_expensive_neighborhood'] = data["Neighborhood Code"].isin(neighborhoods).astype("int64")
    return data


def substitute_roof_material(data):
    """
    Input:
      data (data frame): a data frame containing a 'Roof Material' column.  Its values
                         should be limited to those found in the codebook
    Output:
      data frame identical to the input except with a refactored 'Roof Material' column
    """
    mapping = {1.0:'Shingle/Asphalt',
               2.0:'Tar&Gravel',
               3.0:'Slate',
               4.0:'Shake',
               5.0:'Tile',
               6.0:'Other'}
    return data.replace({'Roof Material':mapping})
def ohe_roof_material(data):
    """
    One-hot-encodes roof material.  New columns are of the form 0x_QUALITY.
    """
    #print(data[['Roof Material']])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = encoder.fit_transform(data[['Roof Material']])
    new_name = ['rfm_'+cat for cat in encoder.categories_[0]]
    ohe_df = pd.DataFrame(encoded,columns=new_name)
    #print(encoded)
    if new_name[0] in data.columns:
        data = data.drop(new_name,axis=1)
    data = pd.concat([data,ohe_df],axis=1)
    return data


# +
def process_data_gm(data, pipeline_functions, prediction_col, test=False):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    if test:
        X = data.to_numpy()
        return X
    else:
        X = data.drop(columns=[prediction_col]).to_numpy()
        y = data.loc[:, prediction_col].to_numpy()
        return X, y

def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]


# -

class ComposedTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.lower_bound = 499
        self.upper_bound = np.inf
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def transform_fit(self, X, y=None):
        pipeline1 = [
            (remove_outliers, ["Sale Price", 499], None),
            (Log_Trans, ["Sale Price"], None),
            (add_total_bedrooms, None, None),
            (select_columns, ['Log Sale Price', 'Bedrooms'], None)
        ]
        return process_data_gm()


def preprocess_train(data):
    pl = [
        (remove_outliers, ["Sale Price",499], None),
        (Log_Trans, ["Sale Price"], None),
        (Log_Trans, ["Building Square Feet"], None),
        (add_total_bedrooms, None, None),
        (select_columns, ['Log Sale Price', 'Bedrooms', 'Log Building Square Feet'], None)
    ]
    return process_data_gm(data, pl, 'Log Sale Price')


def preprocess_test(data):
    pl = [
        (Log_Trans, ["Building Square Feet"], None),
        (add_total_bedrooms, None, None),
        (select_columns, ['Log Sale Price', 'Bedrooms', 'Log Building Square Feet'], None)
    ]
    return process_data_gm(data, pl, 'Log Sale Price', test=True)


def create_pipeline():
    """Create a machine learning pipeline"""
    
    # Define the pipeline
    pipeline = Pipeline([
        ("lr",LinearRegression())
    ])
    
    return pipeline
