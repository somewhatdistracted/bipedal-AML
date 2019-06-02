import tensorflow as tf

model = await tf.loadLayersModel('D:\Biped\preTrainedModel.json')

def getModel():
	return model