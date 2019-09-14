import numpy as np
import functools
import tensorflow as tf

from time import time
from avatar import Avatar
import dynamic_fixed_point as dfxp

avatar = Avatar()

def average_gradients(tower_grads):
    avg_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(tf.stack(grads), axis=0)
        v = grad_and_vars[0][1]
        avg_grads.append((grad, v))
    return avg_grads


def tower_reduce_mean(towers):
    return tf.reduce_mean(tf.stack(towers), axis=0)

def write_file(file_name, num, b, h, w, c):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        for j in range(h):
            for m in range(w):
                for n in range(c):
                    file.write(str('%d:%d:%d:%d : %.6f' % (i, j, m, n, num[i][j][m][n])))
                    file.write('\r\n')

def write_file_fc(file_name, num, b, h):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        for j in range(h):
            file.write(str('%d:%d : %.6f' % (i, j, num[i][j])))
            file.write('\r\n')                  

def write_file_soft(file_name, num, b):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        file.write(str('%.6f' % num[i]))
        file.write('\r\n') 

class LearningRateScheduler:
    def __init__(self, lr, lr_decay_epoch, lr_decay_factor):
        self.lr = tf.get_variable('learning_rate', initializer=lr)
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor
        self.epoch = tf.get_variable('lr_scheduler_step',
            dtype=tf.int32, initializer=0)

        update_epoch = tf.assign(self.epoch, self.epoch+1)
        with tf.control_dependencies([update_epoch]):
            self.update_lr = tf.assign(self.lr, tf.cond(
                tf.equal(tf.mod(self.epoch, self.lr_decay_epoch), 0),
                lambda : self.lr * self.lr_decay_factor,
                lambda : self.lr,
            ))

    def step(self):
        '''
        Op for updating learning rate.

        Should be called at the end of an epoch.
        '''
        return self.update_lr


class Trainer:
    def __init__(self, model, dataset, dataset_name, logger, params):

        self.n_epoch = params.n_epoch
        self.exp_path = params.exp_path

        self.logger = logger

        self.graph = tf.Graph()
        with self.graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            self.lr_scheduler = LearningRateScheduler(params.lr, params.lr_decay_epoch, params.lr_decay_factor)
            optimizer = tf.train.MomentumOptimizer(params.lr, params.momentum)

            tower_grads, tower_loss = [], []

            with tf.variable_scope(tf.get_variable_scope()):

                images = avatar.batch_data()
                images = tf.cast(tf.reshape(images, [-1,32,32,3]), dtype=tf.float32)
                lables = avatar.batch_lable()

                conv1 =  dfxp.Conv2d_q(name='conv1', bits=params.bits, training = False, ksize=[3, 3, 3, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.conv_in_1 = images
                self.batch_in_1, _, _ = conv1.forward(self.conv_in_1)
                batch1 = dfxp.Normalization_q(name='batch1', bits=params.bits, num_features=16, training=True)
                self.scale_in_1 = batch1.forward(self.batch_in_1)
                scale1 = dfxp.Rescale_q(name='scale1', bits=params.bits, training=False, num_features=16)
                self.relu_in_1, self.g, self.b = scale1.forward(self.scale_in_1)
                relu1 = dfxp.ReLU_q()
                self.conv_in_2 = relu1.forward(self.relu_in_1)

                conv2 =  dfxp.Conv2d_q(name='conv2', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_2, self.conv_in_2_q, _ = conv2.forward(self.conv_in_2)
                batch2 = dfxp.Normalization_q(name='batch2', bits=params.bits, num_features=16, training=True)
                self.scale_in_2 = batch2.forward(self.batch_in_2)
                scale2 = dfxp.Rescale_q(name='scale2', bits=params.bits, training=False, num_features=16)
                self.relu_in_2, _, _ = scale2.forward(self.scale_in_2)
                relu2 = dfxp.ReLU_q()
                self.conv_in_3 = relu2.forward(self.relu_in_2) 

                conv3 =  dfxp.Conv2d_q(name='conv3', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_3, _, _ = conv3.forward(self.conv_in_3)
                batch3 = dfxp.Normalization_q(name='batch3', bits=params.bits, num_features=16, training=True)
                self.scale_in_3 = batch3.forward(self.batch_in_3)
                scale3 = dfxp.Rescale_q(name='scale3', bits=params.bits, training=False, num_features=16)
                self.relu_in_32, _, _ = scale3.forward(self.scale_in_3)
                self.relu_in_3 = self.relu_in_32 + self.conv_in_2_q
                relu3 = dfxp.ReLU_q()
                self.conv_in_4 = relu3.forward(self.relu_in_3)  

                conv4 =  dfxp.Conv2d_q(name='conv4', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_4, self.conv_in_4_q, _ = conv4.forward(self.conv_in_4)
                batch4 = dfxp.Normalization_q(name='batch4', bits=params.bits, num_features=16, training=True)
                self.scale_in_4 = batch4.forward(self.batch_in_4)
                scale4 = dfxp.Rescale_q(name='scale4', bits=params.bits, training=False, num_features=16)
                self.relu_in_4, _, _ = scale4.forward(self.scale_in_4)
                relu4 = dfxp.ReLU_q()
                self.conv_in_5 = relu4.forward(self.relu_in_4) 

                conv5 =  dfxp.Conv2d_q(name='conv5', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_5, _, _ = conv5.forward(self.conv_in_5)
                batch5 = dfxp.Normalization_q(name='batch5', bits=params.bits, num_features=16, training=True)
                self.scale_in_5 = batch5.forward(self.batch_in_5)
                scale5 = dfxp.Rescale_q(name='scale5', bits=params.bits, training=False, num_features=16)
                self.relu_in_52, _, _ = scale5.forward(self.scale_in_5)
                self.relu_in_5 = self.relu_in_52 + self.conv_in_4_q
                relu5 = dfxp.ReLU_q()
                self.conv_in_6 = relu5.forward(self.relu_in_5)  

                conv6 =  dfxp.Conv2d_q(name='conv6', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_6, self.conv_in_6_q, _ = conv6.forward(self.conv_in_6)
                batch6 = dfxp.Normalization_q(name='batch6', bits=params.bits, num_features=16, training=True)
                self.scale_in_6 = batch6.forward(self.batch_in_6)
                scale6 = dfxp.Rescale_q(name='scale6', bits=params.bits, training=False, num_features=16)
                self.relu_in_6, _, _ = scale6.forward(self.scale_in_6)
                relu6 = dfxp.ReLU_q()
                self.conv_in_7 = relu6.forward(self.relu_in_6) 

                conv7 =  dfxp.Conv2d_q(name='conv7', bits=params.bits, training = False, ksize=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_7, _, _ = conv7.forward(self.conv_in_7)
                batch7 = dfxp.Normalization_q(name='batch7', bits=params.bits, num_features=16, training=True)
                self.scale_in_7 = batch7.forward(self.batch_in_7)
                scale7 = dfxp.Rescale_q(name='scale7', bits=params.bits, training=False, num_features=16)
                self.relu_in_72, _, _ = scale7.forward(self.scale_in_7)
                self.relu_in_7 = self.relu_in_72 + self.conv_in_6_q
                relu7 = dfxp.ReLU_q()
                self.conv_in_8 = relu7.forward(self.relu_in_7) 
                
                conv8 =  dfxp.Conv2d_q(name='conv8', bits=params.bits, training = False, ksize=[3, 3, 16, 32], strides=[1, 2, 2, 1], padding='VALID')
                self.conv_in_8_1 = tf.pad(self.conv_in_8, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT')
                self.batch_in_8, _, _ = conv8.forward(self.conv_in_8_1)
                batch8 = dfxp.Normalization_q(name='batch8', bits=params.bits, num_features=32, training=True)
                self.scale_in_8 = batch8.forward(self.batch_in_8)
                scale8 = dfxp.Rescale_q(name='scale8', bits=params.bits, training=False, num_features=32)
                self.relu_in_8, _, _ = scale8.forward(self.scale_in_8)
                relu8 = dfxp.ReLU_q()
                self.conv_in_9 = relu8.forward(self.relu_in_8) 

                conv9 =  dfxp.Conv2d_q(name='conv9', bits=params.bits, training = False, ksize=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_9, _, _ = conv9.forward(self.conv_in_9)
                batch9 = dfxp.Normalization_q(name='batch9', bits=params.bits, num_features=32, training=True)
                self.scale_in_9 = batch9.forward(self.batch_in_9)
                scale9 = dfxp.Rescale_q(name='scale9', bits=params.bits, training=False, num_features=32)
                self.relu_in_9, _, _ = scale9.forward(self.scale_in_9)

                conv10 =  dfxp.Conv2d_q(name='conv10', bits=params.bits, training = False, ksize=[1, 1, 16, 32], strides=[1, 2, 2, 1], padding='VALID')
                self.batch_in_10, _, self.w = conv10.forward(self.conv_in_8)
                batch10 = dfxp.Normalization_q(name='batch10', bits=params.bits, num_features=32, training=True)
                self.scale_in_10 = batch10.forward(self.batch_in_10)
                scale10 = dfxp.Rescale_q(name='scale10', bits=params.bits, training=False, num_features=32)
                self.relu_in_102, _, _ = scale10.forward(self.scale_in_10)
                self.relu_in_10 = self.relu_in_102 + self.relu_in_9
                relu10 = dfxp.ReLU_q()
                self.conv_in_11 = relu10.forward(self.relu_in_10) 

                conv11 =  dfxp.Conv2d_q(name='conv11', bits=params.bits, training = False, ksize=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_11, self.conv_in_11_q, _ = conv11.forward(self.conv_in_11)
                batch11 = dfxp.Normalization_q(name='batch11', bits=params.bits, num_features=32, training=True)
                self.scale_in_11 = batch11.forward(self.batch_in_11)
                scale11 = dfxp.Rescale_q(name='scale11', bits=params.bits, training=False, num_features=32)
                self.relu_in_11, _, _ = scale11.forward(self.scale_in_11)
                relu11 = dfxp.ReLU_q()
                self.conv_in_12 = relu11.forward(self.relu_in_11) 

                conv12 =  dfxp.Conv2d_q(name='conv12', bits=params.bits, training = False, ksize=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_12, _, _ = conv12.forward(self.conv_in_12)
                batch12 = dfxp.Normalization_q(name='batch12', bits=params.bits, num_features=32, training=True)
                self.scale_in_12 = batch12.forward(self.batch_in_12)
                scale12 = dfxp.Rescale_q(name='scale12', bits=params.bits, training=False, num_features=32)
                self.relu_in_122, _, _ = scale12.forward(self.scale_in_12)
                self.relu_in_12 = self.relu_in_122 + self.conv_in_11_q
                relu12 = dfxp.ReLU_q()
                self.conv_in_13 = relu12.forward(self.relu_in_12) 

                conv13 =  dfxp.Conv2d_q(name='conv13', bits=params.bits, training = False, ksize=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_13, self.conv_in_13_q, _ = conv13.forward(self.conv_in_13)
                batch13 = dfxp.Normalization_q(name='batch13', bits=params.bits, num_features=32, training=True)
                self.scale_in_13 = batch13.forward(self.batch_in_13)
                scale13 = dfxp.Rescale_q(name='scale13', bits=params.bits, training=False, num_features=32)
                self.relu_in_13, _, _ = scale13.forward(self.scale_in_13)
                relu13 = dfxp.ReLU_q()
                self.conv_in_14 = relu13.forward(self.relu_in_13) 

                conv14 =  dfxp.Conv2d_q(name='conv14', bits=params.bits, training = False, ksize=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_14, _, _ = conv14.forward(self.conv_in_14)
                batch14 = dfxp.Normalization_q(name='batch14', bits=params.bits, num_features=32, training=True)
                self.scale_in_14 = batch14.forward(self.batch_in_14)
                scale14 = dfxp.Rescale_q(name='scale14', bits=params.bits, training=False, num_features=32)
                self.relu_in_142, _, _ = scale14.forward(self.scale_in_14)
                self.relu_in_14 = self.relu_in_142 + self.conv_in_13_q
                relu14 = dfxp.ReLU_q()
                self.conv_in_15 = relu14.forward(self.relu_in_14) 

                conv15 =  dfxp.Conv2d_q(name='conv15', bits=params.bits, training = False, ksize=[3, 3, 32, 64], strides=[1, 2, 2, 1], padding='VALID')
                self.conv_in_15_1 = tf.pad(self.conv_in_15, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT')
                self.batch_in_15, _, _ = conv15.forward(self.conv_in_15_1)
                batch15 = dfxp.Normalization_q(name='batch15', bits=params.bits, num_features=64, training=True)
                self.scale_in_15 = batch15.forward(self.batch_in_15)
                scale15 = dfxp.Rescale_q(name='scale15', bits=params.bits, training=False, num_features=64)
                self.relu_in_15, _, _ = scale15.forward(self.scale_in_15)
                relu15 = dfxp.ReLU_q()
                self.conv_in_16 = relu15.forward(self.relu_in_15) 

                conv16 =  dfxp.Conv2d_q(name='conv16', bits=params.bits, training = False, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_16, _, _ = conv16.forward(self.conv_in_16)
                batch16 = dfxp.Normalization_q(name='batch16', bits=params.bits, num_features=64, training=True)
                self.scale_in_16 = batch16.forward(self.batch_in_16)
                scale16 = dfxp.Rescale_q(name='scale16', bits=params.bits, training=False, num_features=64)
                self.relu_in_16, _, _ = scale16.forward(self.scale_in_16)

                conv17 =  dfxp.Conv2d_q(name='conv17', bits=params.bits, training = False, ksize=[1, 1, 32, 64], strides=[1, 2, 2, 1], padding='VALID')
                self.batch_in_17, _, _ = conv17.forward(self.conv_in_15)
                batch17 = dfxp.Normalization_q(name='batch17', bits=params.bits, num_features=64, training=True)
                self.scale_in_17 = batch17.forward(self.batch_in_17)
                scale17 = dfxp.Rescale_q(name='scale17', bits=params.bits, training=False, num_features=64)
                self.relu_in_172, _, _ = scale17.forward(self.scale_in_17)
                self.relu_in_17 = self.relu_in_172 + self.relu_in_16
                relu17 = dfxp.ReLU_q()
                self.conv_in_18 = relu17.forward(self.relu_in_17) 

                conv18 =  dfxp.Conv2d_q(name='conv18', bits=params.bits, training = False, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_18, self.conv_in_18_q, _ = conv18.forward(self.conv_in_18)
                batch18 = dfxp.Normalization_q(name='batch18', bits=params.bits, num_features=64, training=True)
                self.scale_in_18 = batch18.forward(self.batch_in_18)
                scale18 = dfxp.Rescale_q(name='scale18', bits=params.bits, training=False, num_features=64)
                self.relu_in_18, _, _ = scale18.forward(self.scale_in_18)
                relu18 = dfxp.ReLU_q()
                self.conv_in_19 = relu18.forward(self.relu_in_18) 

                conv19 =  dfxp.Conv2d_q(name='conv19', bits=params.bits, training = False, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_19, _, _ = conv19.forward(self.conv_in_19)
                batch19 = dfxp.Normalization_q(name='batch19', bits=params.bits, num_features=64, training=True)
                self.scale_in_19 = batch19.forward(self.batch_in_19)
                scale19 = dfxp.Rescale_q(name='scale19', bits=params.bits, training=False, num_features=64)
                self.relu_in_192, _, _ = scale19.forward(self.scale_in_19)
                self.relu_in_19 = self.relu_in_192 + self.conv_in_18_q
                relu19 = dfxp.ReLU_q()
                self.conv_in_20 = relu19.forward(self.relu_in_19) 

                conv20 =  dfxp.Conv2d_q(name='conv20', bits=params.bits, training = False, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_20, self.conv_in_20_q, _ = conv20.forward(self.conv_in_20)
                batch20 = dfxp.Normalization_q(name='batch20', bits=params.bits, num_features=64, training=True)
                self.scale_in_20 = batch20.forward(self.batch_in_20)
                scale20 = dfxp.Rescale_q(name='scale20', bits=params.bits, training=False, num_features=64)
                self.relu_in_20, _, _ = scale20.forward(self.scale_in_20)
                relu20 = dfxp.ReLU_q()
                self.conv_in_21 = relu20.forward(self.relu_in_20) 

                conv21 =  dfxp.Conv2d_q(name='conv21', bits=params.bits, training = False, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                self.batch_in_21, _, _ = conv21.forward(self.conv_in_21)
                batch21 = dfxp.Normalization_q(name='batch21', bits=params.bits, num_features=64, training=True)
                self.scale_in_21 = batch21.forward(self.batch_in_21)
                scale21 = dfxp.Rescale_q(name='scale21', bits=params.bits, training=False, num_features=64)
                self.relu_in_212, _, _ = scale21.forward(self.scale_in_21)
                self.relu_in_21 = self.relu_in_212 + self.conv_in_20_q
                relu21 = dfxp.ReLU_q()
                self.conv_in_22 = relu21.forward(self.relu_in_21) 

                pool = dfxp.AvgPool_q(ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
                self.fc_in_1 = pool.forward(self.conv_in_22)

                flat = dfxp.Flatten_q(1*1*64)
                self.flat = flat.forward(self.fc_in_1)

                fc1 = dfxp.Dense_q(name='dense1', bits=params.bits, training = False, in_units=64, units=10)
                self.softmax_in, _ = fc1.forward(self.flat)

                self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=lables, logits=self.softmax_in)

                self.indiff = tf.gradients(self.loss, self.conv_in_2)

                self.train_step = optimizer.minimize(self.loss)

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            self.summary = tf.summary.merge_all()
            self.graph.finalize()

    def train(self):

        self.logger.info('Start of training')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.log_device_placement = False

        with tf.Session(config=config, graph=self.graph) as sess:
            
            self.logger.info('Initializing variables..')
            sess.run(self.init_op)

            for epoch in range(self.n_epoch):
                self.logger.info("*******************************" + str(epoch) + "*******************************")
                
                #sess.run([self.train_step])
                r1 = sess.run([self.indiff])
                #print(np.array(r1).shape)
                write_file('conv_in.txt', r1[0][0], 16, 32, 32, 16)
                sess.run(self.lr_scheduler.step())


