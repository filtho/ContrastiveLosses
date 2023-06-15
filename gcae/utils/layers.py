import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.activations
import tensorflow.keras.layers as layers


class ResidualBlock1(tf.keras.layers.Layer):
    '''
    Define a residual block with pre-activations with batch normalization.

    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock1, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation=None, padding='same'))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation=None, padding='same'))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])

        return x


class ResidualBlock2(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.


    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock2, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation="elu", padding='same'))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation="elu", padding='same'))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])

        return x


class ResidualBlock3(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.


    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock3, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")

        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])

        return x


class ResidualBlock4(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock4, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])

        return x
class ResidualBlock7(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock7, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        #layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        #self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])


        return x
class ResidualBlock8(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock8, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.BatchNormalization())

        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))
        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.BatchNormalization())



        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        #self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])


        return x
class ResidualBlock5(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1", kernel_regularizer=None):
        super(ResidualBlock5, self).__init__()
        self.cname = name
        layers = []

        dim = kernel_size * filters
        dim = 2 *kernel_size* filters ** 2
        limit = tf.math.sqrt(2 * 3 / (dim))

        # layer_args["kernel_size"] * layer_args["filters"]
        k_init = tf.keras.initializers.RandomUniform(-limit, limit)
        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        #self.l0 = layers[0]
        #self.l1 = layers[1]
        #self.l2 = layers[2]
        #self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.LeakyReLU()(tf.keras.layers.Add()([x, input_data]))

        return x


class ResidualBlock6(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1", kernel_regularizer=None):
        super(ResidualBlock6, self).__init__()
        self.cname = name
        layers = []

        dim = kernel_size * filters
        dim = 2 * kernel_size * filters ** 2
        limit = tf.math.sqrt(2 * 3 / (dim))

        # layer_args["kernel_size"] * layer_args["filters"]
        k_init = tf.keras.initializers.RandomUniform(-limit, limit)
        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        #layers.append(LocallyConnected(filters, kernel_size,
        #                               kernel_regularizer=kernel_regularizer, kernel_initializer= k_init))
        #layers.append(tf.keras.layers.BatchNormalization())

        #layers.append(tf.keras.layers.LeakyReLU())
        #print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        #layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.LeakyReLU()(tf.keras.layers.Add()([x, input_data]))

        return x


class ResidualBlockX(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1", kernel_regularizer=None):
        super(ResidualBlockX, self).__init__()
        self.cname = name
        layers = []

        dim = kernel_size * filters

        limit = tf.math.sqrt(2 * 3 / (dim))

        # layer_args["kernel_size"] * layer_args["filters"]
        k_init = tf.keras.initializers.RandomUniform(-limit, limit)

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer))  # , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        # layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))#, kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        # layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))# , kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer))  # , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, input_data])
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPool1D(pool_size=9, strides=4, padding="same")(x)

        return x


class ResidualBlockX_I(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1", kernel_regularizer=None):
        super(ResidualBlockX_I, self).__init__()
        self.cname = name
        layers = []

        dim = kernel_size * filters

        limit = tf.math.sqrt(2 * 3 / (dim))

        # layer_args["kernel_size"] * layer_args["filters"]
        k_init = tf.keras.initializers.RandomUniform(-limit, limit)

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer))  # , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        # layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))#, kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer))  # , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer))  # , kernel_initializer= k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        x = self.layers[0](input_data)
        add = tf.keras.layers.UpSampling1D(size=2)(x)
        x = tf.keras.layers.UpSampling1D(size=2)(x)

        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([x, add])
        x = tf.keras.layers.LeakyReLU()(x)

        return x


class BiasWeightLayerPrime(Layer):

    def __init__(self, **kwargs):
        super(BiasWeightLayerPrime, self).__init__(**kwargs)
        self.primes = []
        for i in range(2, 1000):
            ok = True
            for j in self.primes:
                if i % j == 0:
                    ok = False
                    break
            if ok:
                self.primes.append(i)
        # self.primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
        self.flatten = layers.Flatten()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = (1,) + input_shape[1:]
        weightshape = (len(self.primes), 1000)
        self.kernel = self.add_weight(name='kernel',
                                      shape=weightshape,
                                      initializer='ones',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=weightshape,
                                    initializer='zeros',
                                    trainable=True)
        self.flatten.build(shape)
        super(BiasWeightLayerPrime, self).build(input_shape)  # Be sure to call this at the end

    def createweight(self, x, weights):
        flatx = self.flatten(x)
        # fullrange = tf.range(0, tf.shape(flatx)[1])
        # result = tf.zeros(flatx.shape)
        # for i, prime in enumerate(self.primes):
        # result = tf.reduce_sum(tf.map_fn(lambda vals : tf.gather(weights[vals[0], :], fullrange % vals[1]), tf.convert_to_tensor(list(enumerate(self.primes))), dtype=tf.float32), axis=0)
        result = tf.reduce_sum(tf.map_fn(lambda vals: tf.tile(weights[vals[0], 0:vals[1]],
                                                              tf.truncatediv(tf.shape(flatx)[1:2] + vals[1] - 1,
                                                                             vals[1]))[0:tf.shape(flatx)[1]],
                                         tf.convert_to_tensor(list(enumerate(self.primes))), dtype=tf.float32), axis=0)
        # for j in range(24):
        #    result += tf.where(tf.expand_dims(tf.stop_gradient(tf.map_fn(lambda t : tf.bitwise.bitwise_and(t * prime, 1 << j), fullrange)) > 0, axis=0), weights[i, j], 0.)
        return tf.expand_dims(tf.reshape(result, tf.shape(x)[1:]), axis=0)

    def call(self, x):
        # self.add_loss(tf.math.reduce_sum(1e-5 - tf.math.minimum(tf.square(self.kernel), 1e-5)))
        tf.print("KERNEL", self.kernel[0:1])
        tf.print("BIAS", self.bias[0])
        return x * self.createweight(x, self.kernel) + self.createweight(x, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


class BiasWeightLayer(Layer):

    def __init__(self, **kwargs):
        super(BiasWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = (1,) + input_shape[1:]
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='zeros',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=shape,
                                    initializer='zeros',
                                    trainable=True)
        super(BiasWeightLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.add_loss(tf.math.reduce_sum(1e-5 - tf.math.minimum(tf.square(self.kernel), 1e-5)))
        # tf.print("KERBWL", self.kernel)
        # tf.print("BBWL", self.bias)
        return x * self.kernel + x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class LocallyConnected(Layer):

    def __init__(self, filters, kernel_size, kernel_initializer='glorot_uniform', strides=1, implementation=0,
                 kernel_regularizer=None, **kwargs):
        super(LocallyConnected, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        # TODO ERROR IF NON-1 STRIDE

    def build(self, input_shape, ):
        # Create a trainable weight variable for this layer.
        biasshape = (1,) + input_shape[1:len(input_shape) - 1] + [self.filters]
        kernelshape = (1,) + input_shape[1:len(input_shape) - 1] + [input_shape[-1], self.filters]
        self.kernels = []
        for i in range(self.kernel_size):
            kernel = self.add_weight(name=f'kernel{i}',
                                     shape=kernelshape,
                                     initializer=self.kernel_initializer,
                                     trainable=True,
                                     regularizer=self.kernel_regularizer)
            self.kernels.append(kernel)
        self.bias = self.add_weight(name='bias',
                                    shape=biasshape,
                                    initializer='zeros',
                                    trainable=True)
        # regularizer=self.kernel_regularizer)
        super(LocallyConnected, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x = inputs
        output = tf.expand_dims(self.bias, axis=-2)
        xfactor = tf.expand_dims(x, axis=-2)
        for i, kernel in enumerate(self.kernels):
            step = i // 2
            if i % 2 == 1:
                step *= -1
            output = output + tf.matmul(tf.roll(xfactor, step, axis=-3), kernel)
        output = tf.squeeze(output, axis=-2)
        return output + self.bias

    def compute_output_shape(self, input_shape):
        return (1,) + input_shape[1:len(input_shape) - 1] + [tf.shape(self.bias)[-1]]


class AttentionPoolLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionPoolLayer, self).__init__()
        self.conv1 = layers.Conv1D(**kwargs)
        poolargs = kwargs.copy()
        del poolargs["kernel_size"]
        del poolargs["filters"]
        poolargs["pool_size"] = kwargs["kernel_size"]
        self.avepool = layers.AveragePooling1D(**poolargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = self.compute_output_shape(input_shape)
        self.conv1.build(shape)
        self.avepool.build(shape)
        self._trainable_weights += self.conv1._trainable_weights
        self._trainable_weights += self.avepool._trainable_weights
        super(AttentionPoolLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        weight = tf.math.abs(inputs[:, :, 1::2])
        return self.conv1(inputs[:, :, ::2] * weight) / self.avepool(weight)

    def compute_output_shape(self, input_shape):
        return input_shape[0:-1] + (input_shape[-1] // 2,)


class AttentionEluPoolLayer(Layer):
    """Not ELU, really, softplus..."""

    def __init__(self, bias=0., **kwargs):
        super(AttentionEluPoolLayer, self).__init__()
        self.conv1 = layers.Conv1D(**kwargs)
        self.bias = bias
        poolargs = kwargs.copy()
        del poolargs["kernel_size"]
        del poolargs["filters"]
        poolargs["pool_size"] = kwargs["kernel_size"]
        self.avepool = layers.AveragePooling1D(**poolargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = self.compute_output_shape(input_shape)
        self.conv1.build(shape)
        self.avepool.build(shape)
        self._trainable_weights += self.conv1._trainable_weights
        self._trainable_weights += self.avepool._trainable_weights
        super(AttentionEluPoolLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        weight = tf.keras.activations.softplus(inputs[:, :, 1::2]) + self.bias
        return self.conv1(inputs[:, :, ::2] * weight) / (self.avepool(weight))

    def compute_output_shape(self, input_shape):
        return input_shape[0:-1] + (input_shape[-1] // 2,)


class AttentionConvEmbedding2(Layer):

    def __init__(self, **kwargs):
        super(AttentionConvEmbedding, self).__init__()
        self.conv1 = layers.Conv1D(**kwargs)

        # self.conv1 = tf.keras.layers.LocallyConnected1D(**kwargs)
        poolargs = kwargs.copy()
        del poolargs["kernel_size"]
        del poolargs["filters"]
        poolargs["pool_size"] = kwargs["kernel_size"]
        self.down_sample_ratio = kwargs["strides"]
        self.avepool = layers.AveragePooling1D(**poolargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = self.compute_output_shape(input_shape)
        self.conv1.build(shape)
        self.avepool.build(shape)
        self._trainable_weights += self.conv1._trainable_weights
        self._trainable_weights += self.avepool._trainable_weights
        super(AttentionConvEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        weight = tf.math.abs(inputs)

        tf.shape(inputs)

        return self.conv1(inputs)  # * weight) / self.avepool(weight)

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-2] // self.down_sample_ratio,) + (input_shape[-1],)


class AttentionConvEmbedding(Layer):

    def __init__(self, **kwargs):
        super(AttentionConvEmbedding, self).__init__()
        self.conv1 = layers.Conv1D(**kwargs)

        # self.conv1 = tf.keras.layers.LocallyConnected1D(**kwargs)
        poolargs = kwargs.copy()
        del poolargs["kernel_size"]
        del poolargs["filters"]
        poolargs["pool_size"] = kwargs["kernel_size"]
        self.down_sample_ratio = kwargs["strides"]
        self.avepool = layers.AveragePooling1D(**poolargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        shape = self.compute_output_shape(input_shape)
        self.conv1.build(shape)
        self.avepool.build(shape)
        self._trainable_weights += self.conv1._trainable_weights
        self._trainable_weights += self.avepool._trainable_weights

        len = (input_shape[1])

        pos = tf.cast(tf.range(0, len, 1), tf.float32)
        pos = tf.reshape(pos, [1, len])

        self.p1 = tf.math.cos(tf.cast(pos, tf.float32) / 1000.0 ** (2.0 / len))
        # p2 =  tf.math.cos(tf.cast(pos, tf.float32) / 1000.0 ** (2.0 / 10.0))
        ##p2 = tf.cast(tf.zeros([10,len, 2]),tf.float32)
        # self.pos = tf.concat([pos,p2],axis = -1)
        # pos = tf.concat([p1,p2], axis = 0)

        # tf.repeat(pos,3,axis = 0)
        # pos = tf.concat([pos,pos], axis = 0)
        # tf.print(pos.shape)

        # p2 = tf.cast(tf.zeros([10,len, 2]),tf.float32)
        # self.pos = tf.concat([pos,p2],axis = -1)

        # pos = tf.
        # tf.print(pos.shape)

        super(AttentionConvEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        weight = tf.math.abs(inputs)

        # tf.print(tf.shape(inputs))
        # inputs = tf.keras.layers.Add()([inputs, self.pos])
        # tf.print(tf.shape(self.conv1(inputs)))
        # tf.print(tf.shape(self.p1))
        # tf.squeeze(self.conv1(inputs)) +self.p1

        return tf.squeeze(self.conv1(inputs))  # +self.p1 # self.conv1(inputs)  # * weight) / self.avepool(weight)

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-2] // self.down_sample_ratio,) + (input_shape[-1],)


class AttentionBlock(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added

    should take input from the embedding layer
    '''

    def __init__(self, units, name="AttBlock", kernel_regularizer=None):
        super(AttentionBlock, self).__init__()
        self.cname = name
        layers = []
        # elf.units = units
        ##dim = kernel_size*filters
        #  units = tf.shape(input_data)[1]

        # limit = tf.math.sqrt(2 * 3 / (dim))

        #    #layer_args["kernel_size"] * layer_args["filters"]
        # k_init = tf.keras.initializers.RandomUniform(-limit, limit)
        ## layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))# , kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.D1 = tf.keras.layers.Dense(units)
        self.D2 = tf.keras.layers.Dense(units)
        # self.D3 = tf.keras.layers.Dense(units)

        self.N1 = tf.keras.layers.BatchNormalization()
        self.N2 = tf.keras.layers.BatchNormalization()

        self.N1 = tf.keras.layers.LayerNormalization()
        self.N2 = tf.keras.layers.LayerNormalization()

        # layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))#, kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())

        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.wq = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.wv = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.wk = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))

        self.layers = layers

    def call(self, input_data):
        '''
        Call a residual block.
        :param residual_block: list of layers.py in the block
        :return: output tensor
        '''
        x = tf.squeeze(input_data)
        # x = input_data

        x = self.N1(x)
        Q = self.wq(x)
        V = self.wv(x)
        K = self.wk(x)

        att = tf.keras.layers.Attention()([Q, V, K])

        sum1 = tf.keras.layers.add([x, att])
        sum1 = self.N2(sum1)

        x1 = self.D1(sum1)

        x2 = self.D2(x1)

        # x3 = self.D3(tf.keras.layers.add([x2, sum1]))
        x3 = tf.keras.layers.add([x2, sum1])

        ## print("--- adding {0} ".format(type(self.layers[0])))
        # x = self.layers[0](input_data)
        #
        # for layer in self.layers[1:]:
        #    # print("--- adding {0} ".format(type(layer)))
        #   x = layer(x)

        # # print("--- performing addition ")
        # x = tf.keras.layers.Add()([x, input_data])

        return x3[:, :, tf.newaxis]


class extract_data(tf.keras.layers.Layer):
    '''

    '''

    def __init__(self, units, name="small_dense"):
        super(extract_data, self).__init__()
        self.cname = name
        layers = []

    # self.d1 = tf.keras.layers.Dense(units, activation = "relu")
    # layers.append(tf.keras.layers.Dense(units, activation = "relu"))

    # self.layers = layers

    def call(self, inputs):
        '''


        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = inputs[:, :, 0]  # self.d1(inputs[:,:,0])

        return x

class l2_normalization(tf.keras.layers.Layer):

    def __init__(self, units, name="L2_normalization"):
        super(l2_normalization, self).__init__()
        self.cname = name


    def call(self, inputs):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = tf.keras.backend.l2_normalize(inputs, axis = 1)  # self.d1(inputs[:,:,0])

        return x
class sigmoid(tf.keras.layers.Layer):

    def __init__(self, name="L2_normalization"):
        super(sigmoid, self).__init__()
        self.cname = name


    def call(self, inputs):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''

        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = tf.keras.activations.sigmoid(inputs)  # self.d1(inputs[:,:,0])

        return x

class dense_with_l2_normalization(tf.keras.layers.Layer):

    def __init__(self, units, name="L2_normalization"):
        super(dense_with_l2_normalization, self).__init__()
        self.cname = name

        self.D1 = tf.keras.layers.Dense(units)

    def call(self, inputs):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''
        x = self.D1(inputs)
        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = tf.keras.backend.l2_normalize(x, axis = 1)  # self.d1(inputs[:,:,0])

        return x


class softmax(tf.keras.layers.Layer):

    def __init__(self, units,name="L2_normalization"):
        super(softmax, self).__init__()
        self.cname = name
    def call(self, inputs):

        x = tf.nn.softmax(inputs)

        return x
    



class RN1(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.


    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters_in,  name="res_block1"):
        super(RN1, self).__init__()
        self.cname=name
        layers = []
        reg = tf.keras.regularizers.L2(0.0)

        layers.append(tf.keras.layers.Conv1D(filters_in, 1, padding='same', kernel_regularizer = reg ))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")


        layers.append(tf.keras.layers.Conv1D(filters_in,3, padding='same', kernel_regularizer = reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")


        self.layers = layers
        self.act = tf.keras.layers.Activation('elu')

    def call(self, input_data):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = self.act(tf.keras.layers.Add()([x, input_data]) )


        return x


class RN_down(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.


    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters_in,  name="res_block1"):
        super(RN_down, self).__init__()
        self.cname=name
        layers = []
        reg = tf.keras.regularizers.L2(0.0)

        layers.append(tf.keras.layers.Conv1D(filters_in, 1, strides = 2, padding='same', kernel_regularizer = reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")


        layers.append(tf.keras.layers.Conv1D(filters_in, 3, padding='same', kernel_regularizer = reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")


        self.layers = layers
        self.C1 = tf.keras.layers.Conv1D(filters_in, 1, strides = 2, padding='same', kernel_regularizer = reg)
        self.N1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('elu')

    def call(self, input_data):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        y = self.N1(self.C1(input_data) )


        # print("--- performing addition ")
        x =self.act( tf.keras.layers.Add()([x, y]))
        return x

