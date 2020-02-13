from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class SGLCN(object):
    def __init__(self, placeholders, edge, input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.loss1 = 0
        self.loss2 = 0

        self.inputs = placeholders['features']
        self.edge = edge
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        learning_rate1 = tf.train.exponential_decay(
            learning_rate = FLAGS.lr1, global_step=placeholders['step'], decay_steps=100, decay_rate=0.9, staircase=True)
        learning_rate2 = tf.train.exponential_decay(
            learning_rate = FLAGS.lr2, global_step=placeholders['step'], decay_steps=100, decay_rate=0.9, staircase=True)
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)

        self.layers0 = SparseGraphLearn(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden_gl,
                                        edge=self.edge,
                                        placeholders=self.placeholders,
                                        act=tf.nn.relu,
                                        dropout=True,
                                        sparse_inputs=True)

        self.layers1 = GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden_gcn,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging)

        self.layers2 = GraphConvolution(input_dim=FLAGS.hidden_gcn,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging)
        self.build()
        self.pro = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for var in self.layers0.vars.values():
            self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers1.vars.values():
            self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Graph Learning loss
        D = tf.matrix_diag(tf.ones(self.placeholders['num_nodes']))*-1
        D = tf.sparse_add(D, self.S)*-1
        D = tf.matmul(tf.transpose(self.x), D)
        self.loss1 += tf.trace(tf.matmul(D, self.x)) * FLAGS.losslr1
        self.loss1 -= tf.trace(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S))) * FLAGS.losslr2

        # Cross entropy error
        self.loss2 += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
       
        self.loss = self.loss1 + self.loss2

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def build(self):
        self.x, self.S = self.layers0(self.inputs)

        x1 = self.layers1(self.inputs, self.S)
        self.outputs = self.layers2(x1, self.S)

        # Store model variables for easy access
        self.vars1 = tf.trainable_variables()[0:2]
        self.vars2 = tf.trainable_variables()[2:]

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op1 = self.optimizer1.minimize(self.loss1, var_list=self.vars1)
        self.opt_op2 = self.optimizer2.minimize(self.loss2, var_list=self.vars2)
        self.opt_op = tf.group(self.opt_op1, self.opt_op2)

    def predict(self):
        return tf.nn.softmax(self.outputs)