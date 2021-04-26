import load_data

import argparse
import math
import time
import numpy as np
import tensorflow as tf
# from tensorflow.python.ops.init_ops import he_normal                   #tensorflow==1.10
from tensorflow.contrib.keras.api.keras.initializers import he_normal    #tensorflow==1.13
from tensorflow.contrib.layers.python.layers.layers import batch_norm as batch_norm
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


def command_parser():
    parser = argparse.ArgumentParser(description='parser for FM')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='GradientDescent', help='optimizer')
    parser.add_argument('--weight_threshold', type=float, default=0.1, help='threshold for deciding whether pass to next component')
    parser.add_argument('--lower_bound', type=float, default=math.pow(math.e, -5))
    parser.add_argument('--factor_size', type=int, default=32, help='size of latent feature vector')
    parser.add_argument('--attention_size', type=int, default=32, help='size of attention vector')
    parser.add_argument('--keepprob', type=float, default=0.8, help='keep prob for dropout')
    parser.add_argument('--layer_size', nargs='?', default=[32, 1], help='size of each layer, num_layer==len(layer_size), the last element of layer_size must be 1')
    args = parser.parse_args()
    return args

class ANFM(object):
    def __init__(self, batch_size, epoch, lr, optimizer, weight_threshold, lower_bound,
                 factor_size, attention_size, num_features, keepprob, layer_size):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.optimizer = optimizer
        self.weight_threshold = weight_threshold
        self.lower_bound = lower_bound
        self.factor_size = factor_size
        self.attention_size = attention_size
        self.num_features = num_features
        self.keepprob = keepprob
        self.layer_size = layer_size
        self.num_fields = 5

        self.init_graph()

    def init_weights(self):
        tf.set_random_seed(2020)

        self.X = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input')
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target')
        weights = {}
        #FM weight
        FM_b = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='FM_bias')
        FM_W = tf.Variable(tf.random_uniform(shape=[self.num_features, 1], dtype=tf.float32, minval=-0.01, maxval=0.01), name='linear_coeff')
        # FM_V = tf.Variable(tf.truncated_normal(shape=[self.num_features, self.factor_size], mean=0.0, stddev=0.1), name='quadratic_coeff')
        FM_V = tf.get_variable('quadratic_coeff', shape=[self.num_features, self.factor_size], initializer=he_normal())

        weights['FM'] = {'V':FM_V, 'W':FM_W, 'b':FM_b}

        #DNN weight
        for i in range(len(self.layer_size)):
            if i == 0:
                DNN_b = tf.Variable(tf.constant(0, dtype=tf.float32), name='layer%d_bias'%(i+1))
                # DNN_W = tf.Variable(tf.truncated_normal(shape=[self.factor_size, self.layer_size[i]], dtype=tf.float32, stddev=0.1), name='layer%d_W'%(i+1))
                DNN_W = tf.get_variable('layer%d'%i, shape=[self.factor_size, self.layer_size[i]], initializer=he_normal())
                weights['layer%d'%(i+1)] = {'b':DNN_b, 'W':DNN_W}
            else:
                DNN_b = tf.Variable(tf.constant(0, dtype=tf.float32), name='layer%d_bias'%(i+1))
                # DNN_W = tf.Variable(tf.truncated_normal(shape=[self.layer_size[i-1], self.layer_size[i]], dtype=tf.float32, stddev=0.1), name='layer%d_W'%(i+1))
                DNN_W = tf.get_variable('layer%d'%i, shape=[self.layer_size[i-1], self.layer_size[i]], initializer=he_normal())
                weights['layer%d'%(i+1)] = {'b': DNN_b, 'W': DNN_W}

        #attention layer
        # att_w = tf.Variable(tf.truncated_normal(shape=[self.factor_size, self.attention_size], stddev=0.1))
        att_w = tf.get_variable('att_w', shape=[self.factor_size, self.attention_size], initializer=he_normal())
        att_b = tf.Variable(tf.constant(0.0, dtype=tf.float32))
        att_h = tf.Variable(tf.truncated_normal(shape=[self.attention_size, 1], stddev=0.1))
        weights['attention'] = {'w': att_w, 'b': att_b, 'h': att_h}

        return weights

    def init_graph(self):
        with tf.device('/gpu:0'):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.weights = self.init_weights()
                # ---------------FM-------------------
                self.non_zero_embedding = tf.nn.embedding_lookup(self.weights['FM']['V'], self.X) #None*None*factor_size

                #compute the element_wise_product vectors and attentions simultanously
                element_wise_product = []
                attention = []
                for i in range(self.num_fields-1):
                    for j in range(i+1, self.num_fields):
                        interaction = tf.multiply(self.non_zero_embedding[:,i,:], self.non_zero_embedding[:,j,:])      # None * factor_size
                        att_mul = tf.matmul(interaction, self.weights['attention']['w']) + self.weights['attention']['b']
                        att_mul = tf.nn.relu(att_mul)                             #None * attention_size
                        att = tf.matmul(att_mul, self.weights['attention']['h'])  #None * 1
                        attention.append(att)
                        element_wise_product.append(interaction)

                self.element_wise_product = tf.convert_to_tensor(element_wise_product, dtype=tf.float32) #10 * None * factor_size
                self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2])      #None * 10 * factor_size
                self.attention = tf.convert_to_tensor(attention, dtype=tf.float32)          #10 * None * 1
                self.attention = tf.transpose(self.attention, perm=[1, 0, 2])               #None * 10 * 1
                self._attention_ = tf.nn.softmax(tf.reduce_sum(self.attention, axis=-1))      #None * 10
                self.attention = tf.expand_dims(self._attention_, axis=-1)                    #None * 10 * 1
                self.bi_output = tf.multiply(self.attention, self.element_wise_product)    #None * 10 * factor_size

                #discard some tensor from bi_output(set to zero tensor)
                # indices = tf.where(condition=tf.less(self._attention_, self.weight_threshold), name='indices_to_set_zero') # ? * 2
                # zero_tensors = tf.ones(shape=[self.batch_size * indices.shape[-1], self.factor_size], dtype=tf.float32) * self.lower_bound
                # updates = tf.nn.embedding_lookup(zero_tensors, indices)
                # updates = tf.reduce_sum(updates, axis=1)
                # self.bi_output = tf.tensor_scatter_update(tensor=self.bi_output,
                #                                           indices=indices,
                #                                           updates=updates)

                self.bi_output = tf.reduce_sum(self.bi_output, axis=1)   #None * factor_size

                # ---------------DNN-------------------
                self.output = self.bi_output
                for i in range(len(self.layer_size)):
                    self.output = tf.matmul(self.output, self.weights['layer%d' % (i+1)]['W']) + self.weights['layer%d' % (i+1)]['b']
                    self.output = tf.nn.relu(batch_norm(self.output, fused=False))
                # ---------------DNN-------------------

                self.non_zero_W = tf.nn.embedding_lookup(self.weights['FM']['W'], self.X)
                self.linear_output = tf.reduce_sum(self.non_zero_W, axis=1)
                self.bias = self.weights['FM']['b'] * tf.ones_like(tf.cast(self.Y, dtype=tf.float32))
                self.output = tf.add_n([self.output, self.linear_output, self.bias])
                self.output = tf.sigmoid(self.output)
                # self.loss = tf.contrib.losses.log_loss(self.output, tf.cast(self.Y, tf.float32), weights=1.0, epsilon=1e-07, scope=None)
                # self.loss = tf.losses.log_loss(tf.cast(self.Y, tf.float32), self.output)
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.Y, dtype=tf.float32), logits=self.output)
                self.loss = tf.reduce_sum(self.loss)
                if self.optimizer == 'GradientDescent':
                    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                elif self.optimizer == 'Adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(self.lr)
                elif self.optimizer == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(self.lr)

                self.train_op = self.optimizer.minimize(self.loss)
                self.init = tf.global_variables_initializer()
                # self.sess = tf.Session(config=tf.ConfigProto())
                # self.sess.run(self.init)
                # tf.summary.FileWriter('logdir/', graph=self.graph)

                print('graph initialized')

    def generate_batch(self, data, batch_size):
        index = 0
        batches = len(data['Y']) // batch_size
        for _ in range(1, batches + 1):
            batch_xs = data['X'][index: index + batch_size]
            batch_ys = data['Y'][index: index + batch_size]
            index += batch_size
            yield batch_xs, batch_ys

    def train(self, train_data, val_data, test_data):
        #init eval
        config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=2)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.init)
        init_train_acc, init_train_auc, init_val_acc, init_val_auc, init_test_acc, init_test_auc = self.eval_all(train_data, val_data, test_data)
        print('init_train_acc is {:.5f}, init_val_acc is {:.5f}, init_test_acc is {:.5f}'.format(init_train_acc, init_val_acc, init_test_acc))
        print('init_train_auc is {:.5f}, init_val_auc is {:.5f}, init_test_auc is {:.5f}'.format(init_train_auc, init_val_auc, init_test_auc))

        acc, auc = {}, {}
        train_acc, val_acc, test_acc = [], [], []
        train_auc, val_auc, test_auc = [], [], []
        with tf.gfile.GFile('acc.txt', 'w') as f_acc, tf.gfile.GFile('auc.txt', 'w') as f_auc:
            for epoch in range(self.epoch):
                t1 = time.time()
                _, attention = self.fit(train_data)
                epoch_train_acc, epoch_train_auc, \
                epoch_val_acc, epoch_val_auc, \
                epoch_test_acc, epoch_test_auc = self.eval_all(train_data, val_data, test_data)

                train_acc.append(epoch_train_acc)
                val_acc.append(epoch_val_acc)
                test_acc.append(epoch_test_acc)
                train_auc.append(epoch_train_auc)
                val_auc.append(epoch_val_auc)
                test_auc.append(epoch_test_auc)
                print('epoch {}: train_acc is {:.5f}, val_acc is {:.5f}, test_acc is {:.5f}'.format(epoch + 1, epoch_train_acc, epoch_val_acc, epoch_test_acc))
                print('epoch {}: train_auc is {:.5f}, val_auc is {:.5f}, test_auc is {:.5f}'.format(epoch + 1, epoch_train_auc, epoch_val_auc, epoch_test_auc))
                print(attention)
                print('epoch {}: time consumed is {}'.format(epoch + 1, time.time()-t1))

                acc_line = 'epoch ' + str(epoch + 1) + ' ' + 'train_acc: ' + str(epoch_train_acc) + ', ' + 'val_acc: ' +\
                       str(epoch_val_acc) + ', ' + 'test_acc: ' + str(epoch_test_acc)
                auc_line = 'epoch ' + str(epoch + 1) + ' ' + 'train_auc: ' + str(epoch_train_auc) + ', ' + 'val_auc: ' + \
                       str(epoch_val_auc) + ', ' + 'test_auc: ' + str(epoch_test_auc)
                f_acc.write(acc_line + '\n')
                f_auc.write(auc_line + '\n')
            acc['train'], acc['val'], acc['test'] = train_acc, val_acc, test_acc
            auc['train'], auc['val'], auc['test'] = train_auc, val_auc, test_auc
            # self.plot(self.epoch, acc)
            # self.plot(self.epoch, auc)

    def fit(self, data):
        batches = len(data['Y']) // self.batch_size
        gen = self.generate_batch(data, self.batch_size)
        batch_loss = []
        attention = []
        for batch in range(batches):
            batch_xs, batch_ys = next(gen)
            feed_dict = {self.X:batch_xs, self.Y:[[ys] for ys in batch_ys]}
            _, loss, atten = self.sess.run([self.train_op, self.loss, self._attention_], feed_dict=feed_dict)
            att = self.sess.run(self.attention, feed_dict=feed_dict)
            batch_loss.append(loss)
            if batch == batches-1:
                attention.append(att[0])
        return sum(batch_loss) / batches, attention[0]

    def eval(self, data):
        acc = []
        auc = []
        batches = len(data['Y']) // self.batch_size
        gen = self.generate_batch(data, self.batch_size)
        for _ in range(batches):
            batch_xs, batch_ys = next(gen)
            pred = self.sess.run(self.output, feed_dict={self.X: batch_xs, self.Y: [[ys] for ys in batch_ys]})
            pred = np.reshape(pred, newshape=[pred.shape[0] * pred.shape[1],]).tolist()
            y_pred = list(map(lambda x: math.ceil(x) if x >= 0.5 else math.floor(x), pred))
            y_true = list(map(int, batch_ys))
            acc_score = accuracy_score(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            acc.append(acc_score)
            auc.append(auc_score)
        return sum(acc)/batches, sum(auc)/batches

    def eval_all(self, train_data, val_data, test_data):
        train_acc, train_auc = self.eval(train_data)
        val_acc, val_auc = self.eval(val_data)
        test_acc, test_auc = self.eval(test_data)
        return train_acc, train_auc, val_acc, val_auc, test_acc, test_auc

    def plot(self, epoch, acc):
        x_axis = [i for i in range(1, epoch+1)]
        plt.plot(x_axis, acc['train'])
        plt.plot(x_axis, acc['val'])
        plt.plot(x_axis, acc['test'])
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.show()

def main():
    assistment_file = "../data/assistment2009/assistment.libfm"
    train_file = "../data/assistment2009/assistment.train.libfm"
    test_file = "../data/assistment2009/assistment.test.libfm"
    val_file = "../data/assistment2009/assistment.validation.libfm"

    data = load_data.LoadData(assistment_file, train_file, test_file, val_file)
    args = command_parser()
    anfm = ANFM(args.batch_size, args.epoch, args.lr, args.optimizer, args.weight_threshold, args.lower_bound,
                args.factor_size, args.attention_size, data.num_features, args.keepprob, args.layer_size)
    anfm.train(data.train_data, data.val_data, data.test_data)

if __name__ == '__main__':
    main()
