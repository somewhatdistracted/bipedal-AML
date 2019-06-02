'''
Neural network iterator for
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
import matplotlib.pyplot as plt
import os

def getModel():
    print("Loading Model from JSON...   ", end = "")
    json_file = open("preTrainedModel.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("preTrainedModel_weights.h5")
    print("Loaded.")
    return model

# def permuteModel(model):
#
