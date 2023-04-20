


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
from sklearn.pipeline import FeatureUnion


def remove_outliers(data, variable, degree = 5):
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
    return data[(data[variable]>np.percentile(data[variable],degree)) & (data[variable]<np.percentile(data[variable],100-degree))]


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
    data = substitute_roof_material(data)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = encoder.fit_transform(data[['Roof Material']])
    new_name = ['rfm_'+cat for cat in encoder.categories_[0]]
    ohe_df = pd.DataFrame(encoded,columns=new_name)
    #print(encoded)
    if new_name[0] in data.columns:
        data = data.drop(new_name,axis=1)
    data = pd.concat([data,ohe_df],axis=1)
    return data


def Preprocess(data):
    X = data.copy()
    X = add_in_expensive_neighborhood(X,[44, 93, 94])
    X = add_total_bedrooms(X)
    X = substitute_roof_material(X)
    X = X[["Bedrooms","Building Square Feet","Age Decade"]]
    #print(X)
    return X


def create_pipeline():
    """Create a machine learning pipeline"""
    
    # Define the pipeline
    #pipeline = Pipeline([
    #    ("lr",LinearRegression())
    #])
    ct = ColumnTransformer([
        ('linear_num', "passthrough",["Bedrooms","Age Decade"]),
        ('log_num', FunctionTransformer(np.log), ["Building Square Feet"])
    ])
    pipeline = Pipeline([
        ("columnTrans",ct),
        ("lin-reg",LinearRegression(fit_intercept=True))
    ])
    
    return pipeline


