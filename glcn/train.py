from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import os
import numpy as np

from utils import *
from models import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer'
flags.DEFINE_string('model', 'sglcn', 'Model string.')  # 'gcn', 'glcn', 'sglcn'
flags.DEFINE_float('lr1', 0.005, 'Initial Graph Learning Layer learning rate.')
flags.DEFINE_float('lr2', 0.005, 'Initial Graph Convolution Layer learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_gcn', 30, 'Number of units in GCN hidden layer.')
flags.DEFINE_integer('hidden_gl', 70, 'Number of units in GraphLearning hidden layer.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('losslr1', 0.01, 'xxxx')
flags.DEFINE_float('losslr2', 0.0001, 'xxxx')
flags.DEFINE_integer('seed', 123, 'Number of epochs to train.')
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
adj, edge = preprocess_adj(adj)

# Define placeholders
placeholders = {
    'adj': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_nodes': tf.placeholder(tf.int32),
    'step': tf.placeholder(tf.int32),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = SGLCN(placeholders, edge, input_dim=features[2][1], logging=True)

# Initialize session
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用  
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 1  # 程序最多只能占用指定gpu50%的显存  
config.gpu_options.allow_growth = True      #程序按需申请内存  
sess = tf.Session(config = config)  
# sess = tf.Session()

# Define model evaluation function
def evaluate(features, adj, labels, mask, epoch, placeholders,flag=0):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, adj, labels, mask, epoch, placeholders)
    if flag == 0:
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
    else:
        outs_val = sess.run(model.accuracy, feed_dict=feed_dict_val)
        return outs_val

# Init variables
sess.run(tf.global_variables_initializer())
test_acc_list = []
best_epoch = 0
best = 10000

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, adj, y_train, train_mask, epoch, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # Validation
    cost, acc, duration = evaluate(features, adj, y_val, val_mask, epoch, placeholders)
    test_acc = evaluate(features, adj, y_test, test_mask, epoch, placeholders, flag=1)
    test_acc_list.append(test_acc)


    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))

    if cost < best:
        best_epoch = epoch
        best = cost
        patience = 0

    else:
        patience += 1

    if patience == FLAGS.early_stopping:
        # feed_dict_val = construct_feed_dict(features, adj, y_test, test_mask, epoch, placeholders)
        # Smap = sess.run(tf.sparse_tensor_to_dense(model.S), feed_dict=feed_dict_val)
        # sio.savemat("S.mat", {'adjfix': np.array(Smap)})
        break
 
print("Optimization Finished!")
print("----------------------------------------------")
print("The finall result:", test_acc_list[-101])
print("----------------------------------------------")