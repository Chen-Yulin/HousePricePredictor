from joblib import load
import numpy as np
import pandas as pd 
import os
import re


class Model():
    def __init__(self, dir_name: str) -> None:
        '''
        'dir_name' is the path to your model file on the server, i.e,
        ~/
        The runner will create an instance of the class with path specified.
        Do NOT modify this method.
        '''
        self.dir = dir_name
    def add_in_expensive_neighborhood(self, data, neighborhoods):
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
    def substitute_roof_material(self,data):
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
    def extract_Room(self,description):
        match = re.search(r'(\d+)(?=\s+of which are bedrooms)',description)
        if match:
            return int(match.group(1))
        else:
            return 0

    def add_total_bedrooms(self,data):
        """
        Input:
          data (data frame): a data frame containing at least the Description column.
        """
        with_rooms = data.copy()
        with_rooms['Bedrooms'] = with_rooms['Description'].apply(self.extract_Room)
        return with_rooms
    def Preprocess(self,X):
        X = self.add_in_expensive_neighborhood(X,[44, 93, 94])
        X = self.add_total_bedrooms(X)
        X = self.substitute_roof_material(X)
        X = X[["Bedrooms","Building Square Feet","Age Decade","Garage Indicator","Floodplain",
               "Road Proximity","Sale Year","Repair Condition","Estimate (Building)","Estimate (Land)",
               "Apartments","Wall Material","Basement","Basement Finish"]]
        return X
    def myPredict(self, data: pd.DataFrame) -> np.ndarray:
        '''
        This is the only method that is called by the runner.
        'data' is the pandas Dataframe for the test set.
        It has the same structure as the training data,
        except that it doesn't have the 'Sale Price' column.
        '''
        ## Pre-prossing, if necessary 
        data = self.Preprocess(data)
        ## Load your model 
        ## Remember to prepend your model file with the path 
        ## The following code is just an example, feel free to modify 
        m = load(os.path.join(self.dir, 'm3.joblib.gz'))
        return np.exp(m.predict(data))


