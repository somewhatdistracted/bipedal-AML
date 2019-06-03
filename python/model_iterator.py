'''
Neural network iterator
Ian MacFarlane 2019
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

def norm(x):
    indices = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"]
    n = pd.read_csv("norm.csv", index = indices)
    return (x - n['mean']) / n['std']

def getModel():
    print("Loading Model from JSON...   ", end = "")
    json_file = open("preTrainedModel.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("preTrainedModel_weights.h5")
    print("Loaded.")
    return model

def permuteModel(model):
    weights = np.array(model.get_weights())
    for i in range(weights.shape[0]):
        x, *y = weights[i].shape
        if len(y) == 0:
            delta = np.random.rand(x)
        else:
            z = y[0]
            delta = np.random.rand(x,z)
        weights[i] = weights[i] + delta
    model.set_weights(weights)
    return model

def runModel(model, data):
    t = pd.DataFrame(data, columns = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"])
    return model.predict(norm(t))

#print(runModel(getModel(), [(1,1,1,1,1,1,1,1,1,1,1,1)])) <---- here is the form for data input
