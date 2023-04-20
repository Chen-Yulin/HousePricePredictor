from joblib import load
import numpy as np
import pandas as pd 
import os
import sys
sys.path.append('../Project -part 2/project-p2-train')
from proj import *

class Model():
	def __init__(self, dir_name: str) -> None:
		'''
		'dir_name' is the path to your model file on the server, i.e,
		~/
		The runner will create an instance of the class with path specified.
		Do NOT modify this method.
		'''
		self.dir = dir_name

	def myPredict(self, data: pd.DataFrame) -> np.ndarray:
		'''
		This is the only method that is called by the runner.
		'data' is the pandas Dataframe for the test set.
		It has the same structure as the training data,
		except that it doesn't have the 'Sale Price' column.
		'''
		## Pre-prossing, if necessary 
		data = preprocess_test(data)
		## Load your model 
		## Remember to prepend your model file with the path 
		## The following code is just an example, feel free to modify 
		m = load(os.path.join(self.dir, 'pipeline.joblib.gz'))
		return m.predict(data)




