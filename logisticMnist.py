# 导入所需模块
import tensorflow as tf
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import KFold
import math

# 加载mnist数据集
mnist=tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 显示数据
def show_sample_data(rows, cols):
    """Visualizes individual sample observerations in a grid.
    
    Args:
        rows (int): rows in grid
        cols (int): columns in grid
    """
    fig, axes = plt.subplots(rows, cols, figsize=(rows * 1.5, cols * 1.5))
    for i in range(rows):
        for j in range(cols):
            idx = np.random.randint(len(Y_train))
            axes[i,j].imshow(X_train[idx], cmap='Greys')
            axes[i,j].set_title(('Label:{:d}'.format(Y_train[idx])))
            axes[i,j].set_axis_off()
# 显示数据
# show_sample_data(4, 4)
# plt.show()


# 预处理数据
input_dim = X_train[0].flatten().shape[0]
n_classes = len(np.unique(Y_train))
# Flatten 2D input data X_train and X_test into 1D vector
X_train = X_train.reshape([-1, input_dim])
X_test = X_test.reshape([-1, input_dim])
# Transform labels Y_train and Y_test to one hot encoding
Y_train_onehot = np.zeros([Y_train.shape[0], n_classes])
for i in range(Y_train.shape[0]):
    Y_train_onehot[i][Y_train[i]] = 1
Y_train = Y_train_onehot
Y_test_onehot = np.zeros([Y_test.shape[0], n_classes])
for i in range(Y_test.shape[0]):
    Y_test_onehot[i][Y_test[i]] = 1
Y_test = Y_test_onehot

# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# 建造神经网络
learning_rate = 0.001
layer1_hidden_units = 800
train_keep_rate = 0.8
test_keep_rate = 1.0

tf.reset_default_graph()

# Training data / label ground truth placeholders
X = tf.placeholder(tf.float32, [None, input_dim], name='input_X')
Y = tf.placeholder(tf.float32, [None, n_classes], name='output_Yhat')
keep_rate = tf.placeholder(tf.float32)

# Weights to train
w = {
    'layer_1_w': tf.Variable(tf.truncated_normal([input_dim, layer1_hidden_units]), name='layer_1_w'),
    'layer_2_w': tf.Variable(tf.truncated_normal([layer1_hidden_units, n_classes]), name='layer_2_w'),
}
b = {
    'layer_1_b': tf.Variable(tf.truncated_normal([layer1_hidden_units]), name='layer_1_b'),
    'layer_2_b': tf.Variable(tf.truncated_normal([n_classes]), name='layer_2_b'),
}

# Build Neural Network Graph
# Training Hyperparameters
batch_size = 128
epochs = 50
display_epoch_step = 5
kFolds = 5

# Build the graph
with tf.name_scope('layer_1'):
    layer_1 = tf.add(tf.matmul(X, w['layer_1_w']), b['layer_1_b'])
    layer_1 = tf.nn.relu(layer_1, name='layer_1_relu')
    layer_1 = tf.nn.dropout(layer_1, keep_rate)

with tf.name_scope('layer_2'):
    Yhat_logits = tf.add(tf.matmul(layer_1, w['layer_2_w']), b['layer_2_b'])

# Loss function
train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Yhat_logits), 
                            name='train_loss')

# Optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)


def _accuracy():
    """Calculate accuracy based on Yhat_logits.
    
    Returns:
        (float): ratio of values that matched ground truth labels
    """
    correct_predictions = tf.equal(tf.argmax(Yhat_logits, axis=1), tf.argmax(Y, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def _cross_validate(X_train, Y_train, kFolds):
    """Generator that splits training set into training and validation.  Done via cross-validation.
    
    Args:
        X_train (ndarray): training data 
        Y_train (ndarray): training labels
        kFolds (int): how many folds to split the data set into
    
    Yields:
        X_train[train_idx] (ndarray): training data fold
        Y_train[train_idx] (ndarray): training label fold
        X_train[test_idx] (ndarray): validation data fold
        Y_train[test_idx] (ndarray): validation label fold
    """
    kf = KFold(n_splits=kFolds)
    for train_idx, test_idx in kf.split(X_train):
        yield X_train[train_idx], Y_train[train_idx], X_train[test_idx], Y_train[test_idx]

def _batched(X_train, Y_train, batch_size):
    """Generators that splits training data into batches.
    
    Args:
        X_train (ndarray): training data 
        Y_train (ndarray): training labels
        batch_size (int): batch size
    
    Yields:
        X_train[idx:idx+batch_size] (ndarray): next batch of training data
        Y_train[idx:idx+batch_size] (ndarray): next batch of training label data
    """
    idx = 0
    while idx < len(X_train):
        if len(X_train) - idx < batch_size:
            yield X_train[idx:], Y_train[idx:]
        else:
            yield X_train[idx:idx+batch_size], Y_train[idx:idx+batch_size]
        idx += batch_size

# Train the Model (main loop)
# Run the model
init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
sess = tf.Session()
sess.run(init)

model_train_losses = []
model_train_acc = []
model_valid_losses = []
model_valid_acc = []
model_test_acc = []

for epoch in range(epochs):
# for epoch in tqdm(range(epochs)):
    # Split training data into KFolds
    if epoch % kFolds == 0:
        kFolded = _cross_validate(X_train, Y_train, kFolds)
    X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold = next(kFolded)
    
    # Train using batches
    total_batches = math.ceil(X_train_fold.shape[0] // batch_size)
    batched_data = _batched(X_train_fold, Y_train_fold, batch_size)
    for batch in range(total_batches):
        batch_X_train, batch_Y_train = next(batched_data)
        sess.run(optimizer, feed_dict={X: batch_X_train, Y: batch_Y_train, keep_rate: train_keep_rate})
    
    # Record model performance at each epoch
    e_train_loss, e_train_acc = sess.run([train_loss, _accuracy()], feed_dict={X: X_train, Y: Y_train, keep_rate: train_keep_rate})
    e_valid_loss, e_valid_acc = sess.run([train_loss, _accuracy()], feed_dict={X: X_valid_fold, Y: Y_valid_fold, keep_rate: test_keep_rate})
    e_test_acc = sess.run(_accuracy(), feed_dict={X: X_test, Y: Y_test, keep_rate: test_keep_rate})
    model_train_losses.append(e_train_loss)
    model_train_acc.append(e_train_acc)
    model_valid_losses.append(e_valid_loss)
    model_valid_acc.append(e_valid_acc)
    model_test_acc.append(e_test_acc)

    if epoch % display_epoch_step == 0:    
        # Display during training
        print('-----Epoch: {}-----'.format(epoch+1))
        print('tr_loss\t\t tr_acc \t v_loss\t\t v_acc\t\t test_acc')
        print('{0}\t {1:.4f}\t\t {2:.4f}\t {3:.4f}\t\t {4:.4f}\t\t'
              .format(str(e_train_loss), e_train_acc, e_valid_loss, e_valid_acc, e_test_acc))
    
sess.close()



















#############################################################

# # Build the model of a logistic classifier
# import os
# import gzip
# import six.moves.cPickle as pickle
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.datasets import mnist
# from keras.utils import np_utils

# def build_logistic_model(input_dim, output_dim):
#     model = Sequential()
#     model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

#     return model

# batch_size = 128
# nb_classes = 10
# nb_epoch = 20
# input_dim = 784

# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(60000, input_dim)
# X_test = X_test.reshape(10000, input_dim)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# model = build_logistic_model(input_dim, nb_classes)

# model.summary()

# # compile the model
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, Y_train,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)

# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# # save model as json and yaml
# json_string = model.to_json()
# open('mnist_Logistic_model.json', 'w').write(json_string)
# yaml_string = model.to_yaml()
# open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

# # save the weights in h5 format
# model.save_weights('mnist_Logistic_wts.h5')

# # to read a saved model and weights
# # model = model_from_json(open('my_model_architecture.json').read())
# # model = model_from_yaml(open('my_model_architecture.yaml').read())
# # model.load_weights('my_model_weights.h5')






