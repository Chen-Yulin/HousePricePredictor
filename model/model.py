from joblib import load
import numpy as np
import pandas as pd 
import os

class Model():
	def __init__(self, dir_name: str) -> None:
		'''
		'dir_name' is the path to your model file on the server, i.e,
		~/
		The runner will create an instance of the class with path specified.
		'''
		self.dir = dir_name

	def myPredict(self, data: pd.Dataframe) -> np.ndarray:
		'''
		This is the only method that is called by the runner.
		'data' is the pandas Dataframe for the test set.
		It has the same structure as the training data,
		except that it doesn't have the 'Sale Price' column.
		'''
		## Some feature engineering and pre-prossing ##
		...
		## Load your model ##
		## Remember to prepend your model file with the path ##
		m = load(os.path.join(self.dir, 'your_model_file'))
		...
		
	def preprocess(self, data: pd.Dataframe) -> pd.Dataframe:
		'''
		This method is for preprocessing. 
		This method is not mandatory for the runner,
		it's for your convinience. 
		'''
		## Do your feature engineering and preprocessing ##
		...
	

	## Define your own methods here (inside this class), as you wish ##

	
	
