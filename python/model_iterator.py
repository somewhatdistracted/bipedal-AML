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
    col_names = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ",'PRHipAngleX', 'PRHipAngleY', 'PRHipAngleZ', 'PLHipAngleX', 'PLHipAngleY', 'PLHipAngleZ',
    'PRKneeAngleZ', 'PLKneeAngleZ', "PRAnkleAngleX", "PRAnkleAngleZ", "PLAnkleAngleX", "PLAnkleAngleZ"]
    label_col_names = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"]
    raw = pd.read_csv('newSubjectDataDirection.csv', names = col_names)
    dataset = raw.copy()
    rawlabels = pd.read_csv('newSubjectDataResults.csv', names = label_col_names)
    labelledset = rawlabels.copy()
    dataset = pd.concat([dataset, labelledset], axis = 1)
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.iloc[:, 24:]
    train_dataset = train_dataset.iloc[:, :24]
    test_labels = test_dataset.iloc[:, 24:]
    test_dataset = test_dataset.iloc[:, :24]
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    return (x - train_stats['mean']) / train_stats['std']

    #indices = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    #'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"]
    #n = pd.read_csv("norm.csv", index_col = indices, names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    #return (x - n['mean']) / n['std']

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
            delta = (np.random.rand(x,z) - 0.5) / 1000
        weights[i] = weights[i] + delta
    model.set_weights(weights)
    return model

def runModel(model, data):
    t = pd.DataFrame(data, columns = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ",'PRHipAngleX', 'PRHipAngleY', 'PRHipAngleZ', 'PLHipAngleX', 'PLHipAngleY', 'PLHipAngleZ',
    'PRKneeAngleZ', 'PLKneeAngleZ', "PRAnkleAngleX", "PRAnkleAngleZ", "PLAnkleAngleX", "PLAnkleAngleZ"])
    return model.predict(norm(t))

def getActual(step):
    column_names = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
    'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"]
    raw = pd.read_csv('training_3_sub_1.csv', names = column_names)
    dataset = raw.copy()
    return dataset.iloc[step].tolist()

#print(runModel(getModel(), [(1,1,1,1,1,1,1,1,1,1,1,1)])) <---- here is the form for data input
