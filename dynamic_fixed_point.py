import tensorflow as tf
import numpy as np

def quantize(X, target_overflow_rate, bits, step, training, stochastic=False):

    assert 1 <= bits <= 32, 'invalid value for bits: %d' % bits
    if bits == 32:
        return X

    limit = 2.0 ** (bits - 1)

    X = tf.cast(X, dtype=tf.float32)
    
    @tf.custom_gradient
    def identity(X):
        X = tf.floor(tf.clip_by_value(X / step, -limit, limit-1) + 0.5001) * step
        return X, lambda dy : dy

    if training:
        step_update = update_step(X, target_overflow_rate, bits, step)
        return identity(X), step_update
    else:
        return identity(X), step


def quantize_back(X, target_overflow_rate, bits, step, training, stochastic=True):

    assert 1 <= bits <= 32, 'invalid value for bits: %d' % bits
    if bits == 32:
        return X

    limit = 2.0 ** (bits - 1)

    X = tf.cast(X, dtype=tf.float32)

    @tf.custom_gradient
    def stochastic_identity(X):
        return X, lambda dy : tf.floor(tf.clip_by_value(dy / step, -limit, limit-1) + 0.5) * step

    if training:
        step_update = update_step(X, target_overflow_rate, bits, step)
        return stochastic_identity(X), step_update
    else:
        return stochastic_identity(X), step


def overflow_rate(X, bits, step):

    limit = 2.0 ** (bits - 1)
    X = X / step

    mask_X = tf.cast(tf.greater_equal(X, limit), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit)), tf.float32)
    mask_2X = tf.cast(tf.greater_equal(X, limit/2), tf.float32) + \
        tf.cast(tf.less(X, tf.negative(limit/2)), tf.float32)
    return tf.reduce_mean(mask_X), tf.reduce_mean(mask_2X)


def update_step(X, target_overflow_rate, bits, step):

    overflow_X, overflow_2X = overflow_rate(X, bits, step)
    multiplier = tf.cond(
        overflow_X > target_overflow_rate,
        lambda : 2.0,           
        lambda : tf.cond(
            overflow_2X <= target_overflow_rate,
            lambda : 0.5,       
            lambda : 1.0,       
        )
    )

    return step * multiplier


class Layer_q:
    '''
    Base class for quantized layers.
    '''
    def forward(self, X):
        '''
        Default forward propagation.
        '''
        self.X = X
        self.y = self.X
        return self.y

    def info(self):
        '''
        Returns a one-line description for a quantized layer.
        '''
        return 'quantized layer (default identity)'


class Conv2d_q(Layer_q):
    def __init__(self, name, bits, training, ksize, strides, padding, use_bias=False,
            weight_decay=0, target_overflow_rate=0):

        h, w, Cin, Cout = self.ksize = ksize
        self.strides = strides
        self.padding = padding
        in_units = h * w * Cin
        limit = (6 / in_units) ** 0.5

        self.name = name
        self.train = training
        self.use_bias = use_bias

        step = 2.0 ** -5

        def weight_variable1(h, w, Cin, Cout):
            data = open("weight.txt", 'r')
            imgs = []
            lines = data.readlines()
            num = 0
            for line in lines:
                for db in line.split():
                    imgs.append(float(db))
                    num += 1
                    if num == h * w * Cin * Cout:
                        break
            x = np.array(imgs).astype(np.float32)
            tx = tf.convert_to_tensor(x)
            initial = tf.reshape(tx, [h, w, Cin, Cout])
            return tf.Variable(initial)

        with tf.variable_scope(self.name):
            self.W = weight_variable1(h, w, Cin, Cout)

            self.W_step = tf.get_variable('W_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_bias:
                self.b = tf.get_variable('b', [1, 1, 1, Cout], initializer=tf.zeros_initializer())
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        self.Xq, self.X_step = quantize(self.X, self.target_overflow_rate,
            self.bits, self.X_step, self.train)
        self.Wq, self.W_step = quantize(self.W, self.target_overflow_rate,
            self.bits, self.W_step, self.train)
        self.y = tf.nn.conv2d(self.X, self.W, self.strides, self.padding)

        if self.use_bias:
            self.bq, self.b_step = quantize(self.b, self.target_overflow_rate,
                self.bits, self.b_step, self.train)
            self.y = self.y + self.b

        #self.y, self.grad_step = quantize_back(self.y, self.target_overflow_rate,
                #self.bits, self.grad_step, self.train)

        return self.y, self.X, self.W

    def info(self):
        return '%d bits conv2d: %dx%dx%d stride %dx%d pad %s weight_decay %f' % (
            self.bits, self.ksize[0], self.ksize[1], self.ksize[2],
            self.strides[1], self.strides[2], self.padding, self.weight_decay)


class Dense_q(Layer_q):
    def __init__(self, name, bits, training, in_units, units, use_bias=True, weight_decay=0, target_overflow_rate=0):
 
        limit = (6 / (in_units + units)) ** 0.5
        self.name = name
        self.train = training
        self.use_bias = use_bias

        step = 2.0 ** -5

        def weight_variable2(shape):
            data = open("weight.txt", 'r')
            imgs = []
            lines = data.readlines()
            num = 0
            for line in lines:
                for db in line.split():
                    imgs.append(float(db))
                    num += 1
                    if num == shape:
                        break
            x = np.array(imgs).astype(np.float32)
            tx = tf.convert_to_tensor(x)
            initial = tf.reshape(tx, [12*12*20, 100])
            return tf.Variable(initial)

        def weight_variable3(shape):
            data = open("weight.txt", 'r')
            imgs = []
            lines = data.readlines()
            num = 0
            for line in lines:
                for db in line.split():
                    imgs.append(float(db))
                    num += 1
                    if num == shape:
                        break
            x = np.array(imgs).astype(np.float32)
            tx = tf.convert_to_tensor(x)
            initial = tf.reshape(tx, [64, 10])
            return tf.Variable(initial)

        with tf.variable_scope(self.name):
            if units == 100:
                self.W = weight_variable2(12 * 12 * 20 * 100)
            else:
                self.W = weight_variable3(64 * 10)

            self.W_step = tf.get_variable('W_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_bias:
                self.b = tf.Variable(tf.constant(0.01, shape=[units]))
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        self.Xq, self.X_step = quantize(self.X, self.target_overflow_rate,
            self.bits, self.X_step, self.train)
        self.Wq, self.W_step = quantize(self.W, self.target_overflow_rate,
            self.bits, self.W_step, self.train)
        self.y = tf.matmul(self.X, self.W)

        if self.use_bias:
            self.bq, self.b_step = quantize(self.b, self.target_overflow_rate,
                self.bits, self.b_step, self.train)
            self.y = self.y + self.b

        #self.y, self.grad_step = quantize_back(self.y, self.target_overflow_rate,
                #self.bits, self.grad_step, self.train)

        return self.y, self.W

    def info(self):
        return '%d bits dense: %dx%d weight_decay %f' % (
            self.bits, self.W.shape[0], self.W.shape[1], self.weight_decay)


class Sequential_q(Layer_q):
    def __init__(self, *args):
        self.layers = args

    def forward(self, X):
        self.X = X
        for layer in self.layers:
            X = layer.forward(X)
        self.y = X
        return self.y

    def info(self):
        return '\n\t'.join(['Sequential layer:'] +
            [layer.info() for layer in self.layers])


class Normalization_q(Layer_q):
    def __init__(self, name, bits, num_features, training, momentum=0.9, eps=1e-5, target_overflow_rate=0):
 
        self.name = name
        self.train = training

        step = 2.0 ** -5

        with tf.variable_scope(self.name):
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            self.X_mean_running = tf.get_variable('X_mean_running', [1, 1, 1, num_features],
                initializer=tf.zeros_initializer())
            self.X_var_running = tf.get_variable('X_var_running', [1, 1, 1, num_features],
                initializer=tf.ones_initializer())

        self.eps = eps
        self.momentum = momentum
        self.bits = bits
        self.target_overflow_rate = target_overflow_rate

    def forward(self, X):
        self.X = X

        self.Xq, self.X_step = quantize(self.X, self.target_overflow_rate,
            self.bits, self.X_step, self.train)

        rank = X._rank()
        if rank == 2:
            self.X = tf.expand_dims(self.X, -1)
            self.X = tf.expand_dims(self.X, -1)
        elif rank == 4:
            pass
        else:
            assert False, 'Invalid rank %d' % rank
        self.X_mean_batch, self.X_var_batch = tf.nn.moments(self.X, axes=[0, 1, 2], keep_dims=True)

        if self.train:
            self.X_mean = self.X_mean_batch
            self.X_var = self.X_var_batch

            def update_op(average, variable, momentum):
                return tf.assign(average, momentum * average + (1-momentum) * variable)
            mean_update_op = update_op(self.X_mean_running, self.X_mean_batch, self.momentum)
            var_update_op = update_op(self.X_var_running, self.X_var_batch, self.momentum)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, var_update_op)
        else:
            self.X_mean = self.X_mean_running
            self.X_var = self.X_var_running

        self.y = (self.X - self.X_mean) / ((self.X_var + self.eps) ** 0.5)

        #self.y, self.grad_step = quantize_back(self.y, self.target_overflow_rate,
                #self.bits, self.grad_step, self.train)

        return self.y


class Rescale_q(Layer_q):
    def __init__(self, name, bits, training, num_features, use_beta=True, weight_decay=0,
        target_overflow_rate=0, gamma_initializer=None):

        self.name = name
        self.train = training
        self.use_beta = use_beta

        step = 2 ** -5

        with tf.variable_scope(self.name):
            self.gamma = tf.Variable(tf.constant(0.35, shape=[1, 1, 1, num_features]))

            self.g_step = tf.get_variable('g_step', initializer=step, trainable=False)
            self.X_step = tf.get_variable('X_step', initializer=step, trainable=False)
            self.grad_step = tf.get_variable('grad_step', initializer=step, trainable=False)

            if self.use_beta:
                self.beta = tf.Variable(tf.constant(0.05, shape=[1, 1, 1, num_features]))
                self.b_step = tf.get_variable('b_step', initializer=step, trainable=False)

        self.bits = bits
        self.target_overflow_rate = target_overflow_rate
        self.weight_decay = weight_decay

    def forward(self, X):
        self.X = X

        rank = X._rank()
        if rank == 2:
            self.X = tf.expand_dims(self.X, -1)
            self.X = tf.expand_dims(self.X, -1)
        elif rank == 4:
            pass
        else:
            assert False, 'Invalid rank %d' % rank

        self.Xq, self.X_step = quantize(self.X, self.target_overflow_rate,
            self.bits, self.X_step, self.train)
        self.gq, self.g_step = quantize(self.gamma, self.target_overflow_rate,
            self.bits, self.g_step, self.train)
        self.y = self.X * self.gamma

        if self.use_beta:
            self.bq, self.b_step = quantize(self.beta, self.target_overflow_rate,
                self.bits, self.b_step, self.train)
            self.y = self.y + self.beta

        #self.y, self.grad_step = quantize_back(self.y, self.target_overflow_rate,
        #       self.bits, self.grad_step, self.train)

        return self.y, self.gamma, self.beta


class BatchNorm_q(Sequential_q):
    def __init__(self, name, bits, num_features, training, momentum=0.9, eps=1e-5,
        use_beta=True, weight_decay=0, target_overflow_rate=0, gamma_initializer=None):
  
        self.bits = bits

        super().__init__(
            Normalization_q(
                name=name+'-norm',
                bits=self.bits,
                num_features=num_features,
                training=training,
                momentum=momentum,
                eps=eps,
                target_overflow_rate=target_overflow_rate,
            ),
            Rescale_q(
                name=name+'-rescale',
                bits=self.bits,
                training=training,
                num_features=num_features,
                use_beta=use_beta,
                weight_decay=weight_decay,
                target_overflow_rate=target_overflow_rate,
                gamma_initializer=gamma_initializer,
            )
        )

    def info(self):
        return '%d bits BatchNorm' % self.bits


class ReLU_q(Layer_q):
    def forward(self, X):
        self.X = X
        self.y = tf.maximum(0.0, self.X)
        return self.y

    def info(self):
        return 'ReLU'


class AvgPool_q(Layer_q):
    def __init__(self, ksize, strides, padding):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def forward(self, X):
        self.X = X
        self.y = tf.nn.avg_pool(self.X, self.ksize, self.strides, self.padding)
        return self.y

    def info(self):
        return 'avg pool: %dx%d stride %dx%d' % (
            self.ksize[1], self.ksize[2], self.strides[1], self.strides[2])


class Flatten_q(Layer_q):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, X):
        self.X = X
        self.y = tf.reshape(X, [-1, self.dim])
        return self.y

    def info(self):
        return 'flatten'