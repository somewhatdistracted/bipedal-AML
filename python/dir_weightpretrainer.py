'''
Neural network pretrainer for bipedal walking system.
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

#Disable TensorFlow Build Warnings... don't know the code for all of the annoying deprecated ones
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None

# Dataset Init
print("Loading (24) Dataset:\n\n\n")
col_names = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ",'PRHipAngleX', 'PRHipAngleY', 'PRHipAngleZ', 'PLHipAngleX', 'PLHipAngleY', 'PLHipAngleZ',
'PRKneeAngleZ', 'PLKneeAngleZ', "PRAnkleAngleX", "PRAnkleAngleZ", "PLAnkleAngleX", "PLAnkleAngleZ"]

label_col_names = ['RHipAngleX', 'RHipAngleY', 'RHipAngleZ', 'LHipAngleX', 'LHipAngleY', 'LHipAngleZ',
'RKneeAngleZ', 'LKneeAngleZ', "RAnkleAngleX", "RAnkleAngleZ", "LAnkleAngleX", "LAnkleAngleZ"]

#training_sub_1
raw = pd.read_csv('newSubjectDataDirection.csv', names = col_names)
dataset = raw.copy()
rawlabels = pd.read_csv('newSubjectDataResults.csv', names = label_col_names)
labelledset = rawlabels.copy()

#Pop Time
#dataset.pop("Time") deprecated without time
print(dataset);
print("\nDataset types:")
print(dataset.dtypes)

print("Generating Label Data\n\n\n")
print(labelledset)
print("\n\n")

dataset = pd.concat([dataset, labelledset], axis = 1)
print(dataset)

print("\n\n\nDataset Loaded.\n\n\n")

# Config Train/test
print("Splitting Dataset into Train and Test...");
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print("Splitting X and Y...\n")
train_labels = train_dataset.iloc[:, 24:] #12 for without direction
train_dataset = train_dataset.iloc[:, :24]
test_labels = test_dataset.iloc[:, 24:]
test_dataset = test_dataset.iloc[:, :24]
print(train_dataset)
print(train_labels)

print("\n\n\nDataset Metrics:\n\n\n")
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats.to_csv("norm.csv")
print(train_stats)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

print("\nNormalizing...")
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#Model Config
print("Configuring Model...")
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(512, activation = 'relu', input_shape = [len(train_dataset.keys())]),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(12)
])

#Compile Model
optimizer = tf.keras.optimizers.Adam(0.005)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

# Compile Model
# model.compile(optimizer = tf.train.AdamOptimizer(0.01), loss = 'mse', metrics = ['mae'])
# model.fit(dataset, epochs=10, steps_per_epoch=3);

print("\n\n\nModel Summary:\n\n\n")
print(model.summary())
print("\n\n\n")


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('\nTraining', end='')
        print('.', end='')

EPOCHS = 1000

print("Training Model...")
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

print("\n\n\nModel Trained.\n\n\n")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Degrees]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Degrees^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

#Plot Model Performance over Training
plot_history(history)

#Running on Test Model
print("Running on Test Model...")
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing Set Mean Squared Error: {:5.8f} dTheta".format(mse))

#Save model to file
save = True
if (save):
    print("Saving Model to Disk as: preTrainedModel.json, preTrainedModel_weights.h5")
    model_json = model.to_json()
    with open("preTrainedModel.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("preTrainedModel_weights.h5")





'''
print("Testing on multiple time steps...")
example_batch = normed_train_data[:1]
print("\n\n\nBase\n\n\n")
print(example_batch)


step = pd.DataFrame.from_dict(model.predict(example_batch))
step.rename(columns = {0 : 'RHipAngleX', 1 : 'RHipAngleY', 2 : 'RHipAngleZ', 3 : 'LHipAngleX', 4 : 'LHipAngleY', 5 : 'LHipAngleZ',
6 : 'RKneeAngleZ', 7 : 'LKneeAngleZ', 8 : "RAnkleAngleX", 9 : "RAnkleAngleZ", 10 : "LAnkleAngleX", 11 : "LAnkleAngleZ"}, inplace = True)


for i in range(9): #desired steps - 1
    step = pd.DataFrame.from_dict(model.predict(norm(step)))
    step.rename(columns = {0 : 'RHipAngleX', 1 : 'RHipAngleY', 2 : 'RHipAngleZ', 3 : 'LHipAngleX', 4 : 'LHipAngleY', 5 : 'LHipAngleZ',
    6 : 'RKneeAngleZ', 7 : 'LKneeAngleZ', 8 : "RAnkleAngleX", 9 : "RAnkleAngleZ", 10 : "LAnkleAngleX", 11 : "LAnkleAngleZ"}, inplace = True)

print("\n\n\nAfter 10 Steps:\n\n\n")
print(step)
'''






#
#
#
#
# print("\n\n\n3\n\n\n")
# print(example_result3)
# print("\n\n\n4\n\n\n")
# print(example_result4)



# sess = tf.InteractiveSession
#
# X = tf.placeholder(tf.float32, shape=[None, 12])
# Y = tf.placeholder(tf.float32, shape=[None, 12])
#
# W1 = tf.Variable(tf.zeroes([100,12]))
# b1 = tf.Variable(tf.zeroes([100]))
#
# sess.run(tf.global_variables_initializer())
#
# yhat = tf.matmul(x,W1) + b1
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for _ in range(1000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
