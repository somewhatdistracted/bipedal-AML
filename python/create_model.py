import tensorflow as tf
import pandas as pd

def getModel():
	model = tf.loadLayersModel('D:\Biped\preTrainedModel.json')
	return model