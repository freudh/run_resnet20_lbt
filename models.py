import numpy as np
import tensorflow as tf

import dynamic_fixed_point as dfxp

class Model:
    def __init__(self, bits, dropout, weight_decay, stochastic, training):
        self.bits = bits
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.stochastic = stochastic
        self.training = training
        self.layers = self.get_layers()

    def get_layers(self):
        return []

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        logit = X
        pred = tf.argmax(logit, axis=1, output_type=tf.int32)
        return logit, pred

    def grads_and_vars(self):
        res = []
        for layer in self.layers:
            res += layer.grads_and_vars()
        return res

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.stochastic)
        return grad

    def info(self):
        return '\n'.join([layer.info() for layer in self.layers])


class MNIST_Model(Model):
    def __init__(self, bits, dropout=0.5, weight_decay=0, stochastic=False, training=False):
        super().__init__(bits, dropout, weight_decay, stochastic, training)

    def get_layers(self):
        return [
            dfxp.Conv2d_q(
                name='conv',
                bits=self.bits,
                training = self.training,
                ksize=[5, 5, 1, 20],
                strides=[1, 1, 1, 1],
                padding='VALID',
                weight_decay=self.weight_decay,
            ),
            dfxp.BatchNorm_q(
                name='batch_normolization',
                bits=self.bits,
                num_features=20,
                training=self.training,
                weight_decay=self.weight_decay,
            ),
            dfxp.ReLU_q(),
            dfxp.MaxPool_q(
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID'
            ),
            dfxp.Flatten_q(12*12*20),
            dfxp.Dense_q(
                name='dense1',
                bits=self.bits,
                training = self.training,
                in_units=12*12*20,
                units=100,
                weight_decay=self.weight_decay,
            ),
            dfxp.Dense_q(
                name='dense2',
                bits=self.bits,
                training = self.training,
                in_units=100,
                units=10,
                weight_decay=self.weight_decay,
            )
        ]