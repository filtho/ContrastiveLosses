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

        # layers.append(tf.keras.layers.BatchNormalization())
        # print("--- batch normalization")
        self.norm = tf.keras.layers.BatchNormalization()

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
        x = self.norm(x)
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
        # layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
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
        # self.l3 = layers[3]
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


class ResidualBlock9(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock9, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))

        layers.append(tf.keras.layers.LeakyReLU())
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        layers.append(tf.keras.layers.BatchNormalization())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))
        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.BatchNormalization())

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU()

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
        x = self.act(x)
        x = self.norm(x)
        return x


class ResidualBlock10(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1", trainable=True):
        super(ResidualBlock10, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same', trainable=trainable))
        layers.append(tf.keras.layers.Activation("gelu"))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same', trainable=trainable))
        layers.append(tf.keras.layers.Activation("gelu"))
        layers.append(tf.keras.layers.BatchNormalization())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))
        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same', trainable=trainable))
        # layers.append(tf.keras.layers.BatchNormalization())

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        self.norm = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.act = tf.keras.layers.Activation("gelu")

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
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock10d(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock10d, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same'))

        layers.append(tf.keras.layers.Activation("silu"))
        # layers.append(tf.keras.layers.BatchNormalization())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format( filters//4, kernel_size))

        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size= 1))
        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same'))
        # layers.append(tf.keras.layers.BatchNormalization())

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        # self.norm = tf.keras.layers.BatchNormalization()
        self.c1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.act = tf.keras.layers.Activation("silu")

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
        x = tf.keras.layers.Add()([x, self.c1(input_data)])
        x = self.act(x)
        # x = self.norm(x)
        return x


class ResidualBlock10e(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock10e, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same'))

        layers.append(tf.keras.layers.Activation("silu"))
        # layers.append(tf.keras.layers.BatchNormalization())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format( filters//4, kernel_size))

        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size= 1))
        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same'))
        # layers.append(tf.keras.layers.BatchNormalization())

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        # self.norm = tf.keras.layers.BatchNormalization()
        self.c1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.act = tf.keras.layers.Activation("silu")

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
        x = tf.keras.layers.Add()([x, self.c1(input_data)])
        x = self.act(x)
        # x = self.norm(x)
        return x


class ResidualBlock12(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock12, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU()

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
        x = self.act(x)
        x = self.norm(x)
        return x


class ResidualBlock13(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock13, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())

        layers.append(tf.keras.layers.Dense(filters))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU()

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
        x = self.act(x)
        x = self.norm(x)
        return x


class ResidualBlock11(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.
    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    # This version should be more similar to the resnet blocks, with batch norm and the same sort of ordering of the activations
    def __init__(self, filters, kernel_size, name="res_block1"):
        super(ResidualBlock11, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))

        layers.append(tf.keras.layers.LeakyReLU())
        # layers.append(tf.keras.layers.BatchNormalization())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        layers.append(tf.keras.layers.LeakyReLU())
        # layers.append(tf.keras.layers.BatchNormalization())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))
        layers.append(tf.keras.layers.Conv1D(filters, kernel_size, padding='same'))
        # layers.append(tf.keras.layers.LayerNormalization())

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        # self.l3 = layers[3]
        self.layers = layers
        # self.norm = tf.keras.layers.LayerNormalization()
        self.act = tf.keras.layers.LeakyReLU()

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
        x = self.act(x)
        # x = self.norm(x)
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
        dim = 2 * kernel_size * filters ** 2
        limit = tf.math.sqrt(2 * 3 / (dim))

        # layer_args["kernel_size"] * layer_args["filters"]
        k_init = tf.keras.initializers.RandomUniform(-limit, limit)
        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())

        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        # self.l0 = layers[0]
        # self.l1 = layers[1]
        # self.l2 = layers[2]
        # self.l3 = layers[3]
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
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.LeakyReLU())
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        # layers.append(LocallyConnected(filters, kernel_size,
        #                               kernel_regularizer=kernel_regularizer, kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())

        # layers.append(tf.keras.layers.LeakyReLU())
        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        layers.append(LocallyConnected(filters, kernel_size,
                                       kernel_regularizer=kernel_regularizer, kernel_initializer=k_init))
        layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())
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

        self.D1 = tf.keras.layers.Dense(units, activation="relu")
        self.D2 = tf.keras.layers.Dense(units, activation="relu")
        # self.D3 = tf.keras.layers.Dense(units)

        self.N1 = tf.keras.layers.LayerNormalization()
        self.N2 = tf.keras.layers.LayerNormalization()
        self.N3 = tf.keras.layers.LayerNormalization()

        # layers.append(LocallyConnected(filters, kernel_size, kernel_regularizer = kernel_regularizer))#, kernel_initializer= k_init))
        # layers.append(tf.keras.layers.BatchNormalization())
        # layers.append(tf.keras.layers.LeakyReLU())

        # print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

        self.wq = tf.keras.layers.Dense(units, use_bias=False, activation="relu")
        self.wv = tf.keras.layers.Dense(units, use_bias=False, activation="relu")
        self.wk = tf.keras.layers.Dense(units, use_bias=False, activation="relu")

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
        x2 = self.N3(x2)

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
        x = tf.keras.backend.l2_normalize(inputs, axis=1)  # self.d1(inputs[:,:,0])

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

    def __init__(self, units, name="L2_normalization", dtype=tf.float32, use_bias=True):
        super(dense_with_l2_normalization, self).__init__()
        self.cname = name

        self.D1 = tf.keras.layers.Dense(units, use_bias=use_bias, dtype=dtype,
                                        activity_regularizer=L2Regularizer(l2=1.))

    def call(self, inputs):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''
        x = self.D1(inputs)
        # tf.print(tf.reduce_sum(x**2, axis = 1))
        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = tf.keras.backend.l2_normalize(x, axis=1)  # self.d1(inputs[:,:,0])

        return x


@tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=0.):
        self.l2 = l2

    def __call__(self, x):
        return self.l2 * tf.math.reduce_sum(tf.math.square(x))

    def get_config(self):
        return {'l2': float(self.l2)}


class dense_with_l2_normalization_noise(tf.keras.layers.Layer):

    def __init__(self, units, noise, name="L2_normalization", dtype=tf.float32):
        super(dense_with_l2_normalization_noise, self).__init__()
        self.cname = name

        self.D1 = tf.keras.layers.Dense(units, dtype=dtype, activity_regularizer=L2Regularizer(l2=0.1))
        self.Noise = tf.keras.layers.GaussianNoise(noise, dtype=dtype)

    def call(self, inputs):
        '''
        Call a residual block.

        :param residual_block: list of layers.py in the block
        :return: output tensor

        '''
        x = self.D1(inputs)
        # tf.print(x)
        x = self.Noise(x)
        # tf.print(x)

        # print("--- adding {0} ".format(type(self.layers[0])))
        # input2 = input_data[:,:,0]
        x = tf.keras.backend.l2_normalize(x, axis=1)  # self.d1(inputs[:,:,0])

        return x


class softmax(tf.keras.layers.Layer):

    def __init__(self, units, name="L2_normalization"):
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

    def __init__(self, filters_in, name="res_block1"):
        super(RN1, self).__init__()
        self.cname = name
        layers = []
        reg = tf.keras.regularizers.L2(0.0)

        layers.append(tf.keras.layers.Conv1D(filters_in, 1, padding='same', kernel_regularizer=reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")

        layers.append(tf.keras.layers.Conv1D(filters_in, 3, padding='same', kernel_regularizer=reg))
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
        x = self.act(tf.keras.layers.Add()([x, input_data]))

        return x


class RN_down(tf.keras.layers.Layer):
    '''
    Define a residual block with conv act bn.


    :param filters:
    :param kernel_size:
    :return: list of layers.py added
    '''

    def __init__(self, filters_in, name="res_block1"):
        super(RN_down, self).__init__()
        self.cname = name
        layers = []
        reg = tf.keras.regularizers.L2(0.0)

        layers.append(tf.keras.layers.Conv1D(filters_in, 1, strides=2, padding='same', kernel_regularizer=reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))
        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")

        layers.append(tf.keras.layers.Conv1D(filters_in, 3, padding='same', kernel_regularizer=reg))
        print("--- conv1d  filters: {0} kernel_size: {1}".format(filters_in, 1))

        layers.append(tf.keras.layers.BatchNormalization())
        print("--- batch normalization")
        layers.append(tf.keras.layers.Activation('elu'))
        print("--- elu")

        self.layers = layers
        self.C1 = tf.keras.layers.Conv1D(filters_in, 1, strides=2, padding='same', kernel_regularizer=reg)
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

        y = self.N1(self.C1(input_data))

        # print("--- performing addition ")
        x = self.act(tf.keras.layers.Add()([x, y]))
        return x


class S1_B2(tf.keras.layers.Layer):

    def __init__(self, name="res_block1"):
        """ A short implementation for a residual block"""

        super(S1_B2, self).__init__()
        self.cname = name
        layers = []

        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        self.layers = layers
        self.act = tf.keras.layers.Activation('relu')

    def call(self, input_data):
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")
        x = tf.keras.layers.Add()([input_data, x])
        x = self.act(x)

        return x


class S1_B1(tf.keras.layers.Layer):

    def __init__(self, name="res_block1"):
        """ A short implementation for a residual block"""

        super(S1_B1, self).__init__()
        self.cname = name
        layers = []
        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        self.layers = layers

        self.C1 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same")
        self.N1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, input_data):
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")

        y = self.C1(input_data)
        y = self.N1(y)

        x = tf.keras.layers.Add()([y, x])
        x = self.act(x)

        return x


class S1_B3(tf.keras.layers.Layer):

    def __init__(self, filters, name="res_block1"):
        """ A short implementation for a residual block"""

        super(S1_B3, self).__init__()
        self.cname = name
        layers = []
        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=3, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())

        self.layers = layers

        # self.C1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")
        # self.N1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, input_data):
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")

        # y = self.C1(input_data)
        # y = self.N1(y)

        x = tf.keras.layers.Add()([input_data, x])
        # x = self.act(x)

        return x


class S1_B4(tf.keras.layers.Layer):

    def __init__(self, filters, name="res_block1"):
        """ A short implementation for a residual block"""

        super(S1_B4, self).__init__()
        self.cname = name
        self.filters = filters
        layers = []
        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=3, strides=2, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))

        layers.append(tf.keras.layers.Conv1D(filters=filters, kernel_size=3, strides=1, padding="same"))
        layers.append(tf.keras.layers.BatchNormalization())

        self.layers = layers

        self.C1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=2, padding="same")
        self.N1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        # self.pool = tf.keras.layers.AveragePooling1D(pool_size = 2, padding = "same")

    def call(self, input_data):
        x = self.layers[0](input_data)
        for layer in self.layers[1:]:
            # print("--- adding {0} ".format(type(layer)))
            x = layer(x)

        # print("--- performing addition ")

        y = self.C1(input_data)
        y = self.N1(y)
        y = self.act(y)
        # y = self.pool(input_data)
        # y = tf.concat([y,y], axis = -1)
        x = tf.keras.layers.Add()([y, x])
        x = self.act(x)

        return x


class test_layer(tf.keras.layers.Layer):

    def __init__(self, name="sparse", sparsifies=[0.5]):
        """ A short implementation for a residual block"""

        super(test_layer, self).__init__()
        self.cname = name

        self.sparsifies = sparsifies  # [0.8, 0.85,0.9,0.95]
        self.sparsify_input = True
        self.project = False
        self.missing_val = -1

    def call(self, inputs, training=True):
        x = inputs
        if training:
            x = tf.concat([x, x], axis=0)

        if self.sparsify_input and training:

            sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
            noise_fraction = 0.05
        else:
            sparsify_fraction = 0.0
            noise_fraction = 0.0

        try:
            if self.project:
                sparsify_fraction = 0.0
        except:
            pass

        num_samples = tf.shape(x)[0]
        n_markers = tf.shape(x)[1]
        missing_value = self.missing_val

        # mask = tf.experimental.numpy.full(shape=(num_samples, n_markers), fill_value=1.0,
        # dtype=tf.float32)
        # mask = inputs[:,:,1]

        # b = tf.sparse.SparseTensor(indices=indices,
        #                          values=(tf.repeat(-1.0, tf.shape(indices)[0])),
        #                         dense_shape=(num_samples, n_markers))
        # mask = tf.sparse.add(mask, b)
        # data = inputs[:,:,0]
        # mask = inputs[:,:,1]
        b = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1)
        mask = tf.where(b < sparsify_fraction, 0., 1.)

        # mask = tf.tensor_scatter_nd_update(inputs[:,:,1], indices, missing_value *tf.ones(shape = tf.shape(indices)[0]), name=None)
        # sparsified_data = tf.tensor_scatter_nd_update(inputs[:,:,0], indices, missing_value *tf.ones(shape = tf.shape(indices)[0]), name=None)

        # assert tf.shape(x[:,:,1,tf.newaxis]) == tf.shape(mask[:,:,tf.newaxis])
        # assert tf.shape(mask[:,:,tf.newaxis]) == tf.shape(sparsified_data[:,:,tf.newaxis])
        b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        noise_mask = tf.cast(tf.where(b_noise < noise_fraction, 1.0, 0.), x.dtype)
        # noised_data = tf.math.abs(x[:,:,0]-0.5*noise_mask)
        # noised_data = tf.math.floormod(x[:,:,0]*2. + 1.*noise_mask,3)/2.

        assert x.dtype == tf.float16
        assert ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2).dtype == tf.float16
        assert noise_mask.dtype == tf.float16
        # bb = x[:,:,0]  -((x[:,:,0]*2 -1)*0.5 +tf.math.floor(b_noise/0.25 * 2+1 - 1.5)/2)*noise_mask
        # noised_data = x[:, :, 0] - ((x[:, :, 0]  - 0.5)  + tf.math.floor(b_noise / 0.5 -0.5) / 2) * noise_mask
        inter = ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2)
        noised_data = x[:, :, 0] - inter * noise_mask

        sparsified_data = tf.math.add(tf.math.multiply(noised_data, mask), -1 * missing_value * (mask - 1))

        # sparsified_data = tf.keras.layers.normalization

        input_data_train = tf.stack([tf.squeeze(sparsified_data[:, :, tf.newaxis]), tf.squeeze(mask[:, :, tf.newaxis]),
                                     tf.squeeze(x[:, :, 1, tf.newaxis])], axis=-1)

        # input_data_train = tf.concat([sparsified_data[:,:,tf.newaxis], mask[:,:,tf.newaxis], x[:,:,1,tf.newaxis]], axis=-1)

        # original_data_mask = tf.experimental.numpy.full(shape=(tf.shape(original_genotypes)[0], self.n_markers),
        # fill_value=1.0,
        # dtype=tf.float32)
        # original_genotypes = tf.stack([original_genotypes[:, :self.n_markers], original_data_mask], axis=-1)

        return input_data_train  # , x, #inds, original_genotypes, original_inds


class sep_sparse_block(tf.keras.layers.Layer):

    def __init__(self, name="sep_sparse_block", sparsifies=[0.5], noise=0.0):
        """ A short implementation for a residual block"""

        super(sep_sparse_block, self).__init__()
        self.cname = name

        self.sparsifies = sparsifies  # [0.8, 0.85,0.9,0.95]
        self.noise = noise  # [0.8, 0.85,0.9,0.95]
        self.sparsify_input = True
        self.project = False
        self.missing_val = -1

    def sparse_one(self, x, training=True):

        if self.sparsify_input and training:

            sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
            # sparsify_fraction = tf.random.uniform(shape = [1], minval = 0.5, maxval =0.95)
            sparsify_fraction = tf.random.uniform(shape=[1], minval=0.001, maxval=0.95) / 5

            noise_fraction = self.noise
        else:
            sparsify_fraction = 0.0
            noise_fraction = 0.0

        try:
            if self.project:
                sparsify_fraction = 0.0
                # sparsify_fraction = tf.random.shuffle([0.0, 0.9])[0]

        except:
            pass
        sparsify_fraction = tf.cast(sparsify_fraction, x.dtype)
        noise_fraction = tf.cast(noise_fraction, x.dtype)

        # tf.print(sparsify_fraction)
        num_samples = tf.shape(x)[0]
        n_markers = tf.shape(x)[1]
        missing_value = self.missing_val

        b = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        # mask = tf.cast(tf.where(b < sparsify_fraction, 0., 1.),x.dtype)
        mask = tf.cast(tf.where(b < sparsify_fraction, 1, 0), x.dtype)

        # b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1)
        # noise_mask = tf.where(b_noise < noise_fraction, 1.0,0.)

        b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        noise_mask = tf.cast(tf.where(b_noise < noise_fraction, 1.0, 0.), x.dtype)

        # noised_data = x[:, :, 0] - ((x[:, :, 0]  - 0.5)  + -2*(tf.math.abs(x[:, :, 0]  - 0.5)  - 0.5)*tf.math.floor(b_noise / 0.5 -0.5) / 2) * noise_mask

        assert x.dtype == tf.float16
        assert ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2).dtype == tf.float16
        assert noise_mask.dtype == tf.float16
        # bb = x[:,:,0]  -((x[:,:,0]*2 -1)*0.5 +tf.math.floor(b_noise/0.25 * 2+1 - 1.5)/2)*noise_mask
        # noised_data = x[:, :, 0] - ((x[:, :, 0]  - 0.5)  + tf.math.floor(b_noise / 0.5 -0.5) / 2) * noise_mask
        inter = ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2)
        noised_data = x[:, :, 0] - inter * noise_mask

        mask_orig = mask
        mask = mask + tf.concat([tf.zeros((num_samples, 1), dtype=x.dtype), mask_orig[:, :-1]], axis=1)
        mask = mask + tf.concat([tf.zeros((num_samples, 2), dtype=x.dtype), mask_orig[:, :-2]], axis=1)
        mask = mask + tf.concat([tf.zeros((num_samples, 3), dtype=x.dtype), mask_orig[:, :-3]], axis=1)
        mask = mask + tf.concat([tf.zeros((num_samples, 4), dtype=x.dtype), mask_orig[:, :-4]], axis=1)
        mask = mask + tf.concat([tf.zeros((num_samples, 5), dtype=x.dtype), mask_orig[:, :-5]], axis=1)
        mask = tf.cast(tf.where((mask == 0), 1, 0), x.dtype)
        sparsified_data = (tf.math.add(tf.math.multiply(noised_data, mask), -1 * missing_value * (mask - 1)))
        # sparsified_data = (tf.math.add(tf.math.multiply(noised_data, mask), -1 * missing_value * (mask - 1)) +1.) * 1./(1. -sparsify_fraction)

        input_data_train = tf.stack(
            [tf.squeeze(sparsified_data[:, :, tf.newaxis]), tf.squeeze(mask[:, :, tf.newaxis] * -1 + 1),
             tf.squeeze(x[:, :, 1, tf.newaxis])], axis=-1)

        # input_data_train = input_data_train[:,:,:,tf.newaxis]
        # tf.print(input_data_train)
        return input_data_train

    def call(self, inputs, training=True):
        x = inputs

        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, -1)

        if training:
            # x = tf.concat([x, x], axis=0)
            input_data_train = tf.concat([self.sparse_one(x, training), self.sparse_one(x, training)], axis=0)
        else:
            input_data_train = self.sparse_one(x, training)

        return input_data_train  ## , x, #inds, original_genotypes, original_inds


class sep_sparse(tf.keras.layers.Layer):

    def __init__(self, name="sparse", sparsifies=[0.5], noise=0.0, recomb=False, gamma = 0.1):
        """ A short implementation for a residual block"""

        super(sep_sparse, self).__init__()
        self.cname = name
        self.recomb = recomb
        self.sparsifies = sparsifies  # [0.8, 0.85,0.9,0.95]
        self.noise = noise  # [0.8, 0.85,0.9,0.95]
        self.sparsify_input = True
        self.project = False
        self.missing_val = -1
        self.gamma = gamma

    def sparse_one(self, x, training=True):

        if self.sparsify_input and training:

            # sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
            # sparsify_fraction = tf.random.uniform(shape = [1], minval = 0.5, maxval =0.95)
            sparsify_fraction = tf.random.uniform(shape=[1], minval=0.01, maxval=0.3)

            noise_fraction = self.noise
        else:
            sparsify_fraction = 0.0
            noise_fraction = 0.0

        try:
            if self.project:
                sparsify_fraction = 0.0
                # sparsify_fraction = tf.random.shuffle([0.0, 0.9])[0]

        except:
            pass
        sparsify_fraction = tf.cast(sparsify_fraction, x.dtype)
        noise_fraction = tf.cast(noise_fraction, x.dtype)

        # tf.print(sparsify_fraction)
        num_samples = tf.shape(x)[0]
        n_markers = tf.shape(x)[1]
        missing_value = self.missing_val

        b = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        mask = tf.cast(tf.where(b < sparsify_fraction, 0., 1.), x.dtype)

        # b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1)
        # noise_mask = tf.where(b_noise < noise_fraction, 1.0,0.)

        b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        noise_mask = tf.cast(tf.where(b_noise < noise_fraction, 1.0, 0.), x.dtype)

        # noised_data = x[:, :, 0] - ((x[:, :, 0]  - 0.5)  + -2*(tf.math.abs(x[:, :, 0]  - 0.5)  - 0.5)*tf.math.floor(b_noise / 0.5 -0.5) / 2) * noise_mask

        assert x.dtype == tf.float16
        assert ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2).dtype == tf.float16
        assert noise_mask.dtype == tf.float16
        # bb = x[:,:,0]  -((x[:,:,0]*2 -1)*0.5 +tf.math.floor(b_noise/0.25 * 2+1 - 1.5)/2)*noise_mask
        # noised_data = x[:, :, 0] - ((x[:, :, 0]  - 0.5)  + tf.math.floor(b_noise / 0.5 -0.5) / 2) * noise_mask
        inter = ((x[:, :, 0] - 0.5) + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * tf.math.floor(
            b_noise / 0.5 - 0.5) / 2)
        noised_data = x[:, :, 0] - inter * noise_mask * 0

        sparsified_data = (tf.math.add(tf.math.multiply(noised_data, mask), -1 * missing_value * (mask - 1)))
        # sparsified_data = (tf.math.add(tf.math.multiply(noised_data, mask), -1 * missing_value * (mask - 1)) +1.) * 1./(1. -sparsify_fraction)

        input_data_train = tf.stack(
            [tf.squeeze(sparsified_data[:, :, tf.newaxis]), tf.squeeze(mask[:, :, tf.newaxis] * -1 + 1),
             tf.squeeze(x[:, :, 1, tf.newaxis])], axis=-1)

        # input_data_train = input_data_train[:,:,:,tf.newaxis]
        # tf.print(input_data_train)
        return input_data_train

    def call(self, inputs, training=True):
        x = inputs

        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, -1)

        if training:
            if self.recomb == False:
                # x = tf.concat([x, x], axis=0)
                input_data_train = tf.concat([self.sparse_one(x, training), self.sparse_one(x, training)], axis=0)

            if self.recomb == True:
                parent1 = inputs[:, :, 0]
                # parent2 = tf.random.shuffle(inputs[:,:,0])
                parent2 = tf.gather(inputs[:, :, 0], tf.random.shuffle(tf.range(tf.shape(inputs[:, :, 0])[0])))

                # tf.print(tf.shape(tf.concat([parent1, parent2], axis = 0)))

                offspring = tf.transpose(
                    create_offspring_unlooped_CN(tf.transpose(tf.concat([parent1, parent2], axis=0)), tf.shape(inputs)[1], gamma = self.gamma))
                # tf.print(tf.shape(offspring))

                # offspring = create_offspring_unlooped(inputs[:,:,0], tf.shape(inputs)[1])
                offspring = tf.concat([offspring[:, :, tf.newaxis], x[:, :, tf.newaxis, 1]], axis=-1)
                # offspring = tf.concat([offspring[:,:,tf.newaxis], tf.reshape(x[:,:,tf.newaxis,1], tf.shape(offspring[:,:,tf.newaxis])) ], axis = -1 )

                #input_data_train = tf.concat([self.sparse_one(x, training), self.sparse_one(offspring, training)],
                #                             axis=0)                
                input_data_train = tf.concat([self.sparse_one(offspring, training), self.sparse_one(offspring, training)],
                                             axis=0)

        else:
            input_data_train = self.sparse_one(x, training)

        # input_data_train = tf.concat([self.sparse_one(x,training), self.sparse_one(x,training)], axis = 0)

        return input_data_train  ## , x, #inds, original_genotypes, original_inds


class sep_sparse_oh(tf.keras.layers.Layer):

    def __init__(self, name="sparse", max_sparse=0.99, max_noise=0.99, depth=3, recomb=False, gamma = 0.1):
        """ A short implementation for a residual block"""

        super(sep_sparse_oh, self).__init__()
        self.cname = name
        self.recomb = recomb
        # self.sparsifies = sparsifies  # [0.8, 0.85,0.9,0.95]
        # self.noise = noise  # [0.8, 0.85,0.9,0.95]
        self.sparsify_input = True
        self.project = False
        self.missing_val = -1
        self.max_noise = max_noise
        self.max_sparse = max_sparse
        self.depth = depth
        self.gamma = gamma

    @tf.function
    def sparse_one(self, x, training=True):

        if self.sparsify_input and training:

            # sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
            # sparsify_fraction = tf.random.uniform(shape = [1], minval = 0.5, maxval =0.95)
            sparsify_fraction = tf.random.uniform(shape=[1], minval=0.001, maxval=self.max_sparse)
            noise_fraction = tf.random.uniform(shape=[1], minval=0.00, maxval=self.max_noise)
            noise_fraction2 = noise_fraction
            # noise_fraction = self.noise
        else:
            sparsify_fraction = 0.0
            noise_fraction = 0.0
            noise_fraction2 = tf.cast(1.0, x.dtype)
        try:
            if self.project:
                sparsify_fraction = 0.0
                # sparsify_fraction = tf.random.shuffle([0.0, 0.9])[0]
                # noise_fraction2 = noise_fraction
                # noise_fraction2 = tf.cast(1.0, x.dtype)
                # noise_fraction2 = 1.0 # tf.cast(1.0, x.dtype)

        except:
            pass
        orig_missing = tf.where(x[:, :, 0] == -1)

        #tf.print(tf.unique(tf.reshape(x[:, :, 0],[-1])))
        
        sparsify_fraction = tf.cast(sparsify_fraction, x.dtype)
        noise_fraction = tf.cast(noise_fraction, x.dtype)
        noise_fraction2 = noise_fraction
        if noise_fraction2 == 0:
            noise_fraction2 = tf.cast(1.0, x.dtype)
        # tf.print(sparsify_fraction)
        num_samples = tf.shape(x)[0]
        n_markers = tf.shape(x)[1]
        missing_value = self.missing_val

        b = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        mask = tf.cast(tf.where(b < sparsify_fraction, 0., 1.), x.dtype)

        b_noise = tf.random.uniform(shape=(num_samples, n_markers), minval=0, maxval=1, dtype=x.dtype)
        noise_mask = tf.cast(tf.where(b_noise < noise_fraction, 1.0, 0.), x.dtype)

        # This line gives us  +1 or -1 for genotypes 2 and 0, respectively, and for heterozygotes it flip either up or down, should be 50/50
        inter = ((x[:, :, 0] - 0.5) * tf.math.abs(x[:, :, 0] - 0.5) * 2 + -2 * (tf.math.abs(x[:, :, 0] - 0.5) - 0.5) * (
                    tf.math.floor((noise_mask * b_noise / noise_fraction2 - 0.5)) + 0.5) * 2 / 2)

        noised_data = x[:, :, 0] - inter * noise_mask

        noised_data = tf.tensor_scatter_nd_update(noised_data, orig_missing,
                                                  self.missing_val * tf.ones(shape=tf.shape(orig_missing)[0],
                                                                             dtype=x.dtype))
        #tf.print(tf.unique(tf.reshape(noised_data,[-1])))

        sparsified_data = (noised_data * mask * 2 - 1 * missing_value * (mask - 1))
        sparsified_data = tf.tensor_scatter_nd_update(sparsified_data, orig_missing,
                                                      self.missing_val * tf.ones(shape=tf.shape(orig_missing)[0],
                                                                                 dtype=x.dtype))

        #tf.print(sparsified_data)

        sparsified_data = tf.cast(
            tf.one_hot(tf.cast(sparsified_data, tf.int32), depth=self.depth),
            x.dtype)

        input_data_train = tf.stack([tf.reshape(mask, [tf.shape(x)[0], tf.shape(x)[1]]) * -1 + 1,
                                     tf.reshape(x[:, :, 1], [tf.shape(x)[0], tf.shape(x)[1]])], axis=-1)

        input_data_train = tf.concat([sparsified_data, input_data_train], axis=-1)
        #tf.print((input_data_train))
        return input_data_train

    def call(self, inputs, training=True):
        x = inputs

        if training:
            if self.recomb == False:
                # x = tf.concat([x, x], axis=0)
                input_data_train = tf.concat([self.sparse_one(x, training), self.sparse_one(x, training)], axis=0)

            if self.recomb == True:
                parent1 = inputs[:, :, 0]
                parent2 = tf.gather(inputs[:, :, 0], tf.random.shuffle(tf.range(tf.shape(inputs[:, :, 0])[0])))

                offspring = tf.transpose(create_offspring_unlooped_CN(tf.transpose(tf.concat([parent1, parent2], axis=0)), tf.shape(inputs)[1], gamma = self.gamma ))

                offspring = tf.concat([offspring[:, :, tf.newaxis], x[:, :, tf.newaxis, 1]], axis=-1)

                input_data_train = tf.concat([self.sparse_one(offspring, training), self.sparse_one(offspring, training)], axis=0)

        else:
            input_data_train = self.sparse_one(x, training)


        return input_data_train

def create_offspring_unlooped_orig_implementation(parent_array, n_markers):
    # Using this function we create all offspring at once, instead of creating them one at a time as in a previous implementation
    # TODO: Write clearer comments on what I am doing here, now this is pretty raw
    bp_per_cm = 1000000.
    cm_dist = tf.linspace(1., 100e6, n_markers) / bp_per_cm / 100.

    # bim_file = "/home/x_fitho/ContrastiveLosses/ContrastiveLosses_gcae/gcae/Data/dog/dog_filtered.bim"

    # snp_data = np.genfromtxt(bim_file, usecols=(0, 1, 2, 3, 4, 5), dtype=str)[0:n_markers]
    # cm_dist = snp_data[:, 3].astype(int) / bp_per_cm / 100

    recomb_prob = 1 / 2 * (1 - tf.math.exp(-4 * (cm_dist[1:] - cm_dist[:-1])))
    recomb_prob = tf.cast(recomb_prob, tf.float32)
    recomb_prob2 = tf.tile(tf.reshape(recomb_prob, [n_markers - 1, 1]),
                           [1, tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)])

    u = tf.random.uniform(shape=tf.shape(recomb_prob2), minval=0, maxval=1)

    a1 = tf.math.cumsum(tf.cast(u < recomb_prob2, tf.int32), axis=0) % 2
    a = tf.concat([a1, a1[tf.newaxis, -1, :]], axis=0)
    ind0 = tf.where(a == 0)
    ind1 = tf.where(a == 1)

    ind_0 = tf.stack([ind0[:, 0], ind0[:, 1]], axis=1)
    ind_1 = tf.stack([ind1[:, 0], ind1[:, 1] + tf.cast(tf.shape(parent_array)[1] / 2, tf.int64)], axis=1)

    # Here it seems that we recombine not the two closest ones in the parent array, but we recombine sample
    # i with sample i + num_parent_samples/2, so in the case of "manual recombination selection", or a weighted
    # probability one, just add them subsequently onto the end of the parent array at all times

    parent_0_vals = tf.gather_nd(parent_array, ind_0)
    parent_1_vals = tf.gather_nd(parent_array, ind_1)

    st0 = tf.scatter_nd(ind0, parent_0_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
    st1 = tf.scatter_nd(ind1, parent_1_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])

    offspring = st0 + st1

    return offspring


def create_offspring_unlooped(parent_array, n_markers):
    # Using this function we create all offspring at once, instead of creating them one at a time as in a previous implementation
    # TODO: Write clearer comments on what I am doing here, now this is pretty raw
    bp_per_cm = 1000000.
    cm_dist = tf.linspace(1., 100e6, n_markers) / bp_per_cm / 100.

    # bim_file = "/home/x_fitho/ContrastiveLosses/ContrastiveLosses_gcae/gcae/Data/dog/dog_filtered.bim"

    # snp_data = np.genfromtxt(bim_file, usecols=(0, 1, 2, 3, 4, 5), dtype=str)[0:n_markers]
    # cm_dist = snp_data[:, 3].astype(int) / bp_per_cm / 100

    recomb_prob = 1 / 2 * (1 - tf.math.exp(-4 * (cm_dist[1:] - cm_dist[:-1]))) *5
    recomb_prob = tf.cast(recomb_prob, tf.float32)
    recomb_prob2 = tf.tile(tf.reshape(recomb_prob, [n_markers - 1, 1]),
                           [1, tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)])

    u = tf.random.uniform(shape=tf.shape(recomb_prob2), minval=0, maxval=1)

    modulo = 10
    a11 =  tf.math.cumsum(tf.cast(u < recomb_prob2, tf.int32), axis=0) + tf.random.uniform([1,tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)], minval = 0, maxval = modulo, dtype = tf.int32)
    a1 = a11 % modulo
    a = tf.concat([a1, a1[tf.newaxis, -1, :]], axis=0) # append the last value, which has been removed due to the cumsum
    ind0 = tf.where(a < (modulo-1))

    ind1 = tf.where(a == (modulo-1))
    #tf.print(tf.shape(ind1)[0] / tf.shape(ind0)[0])


    #tf.print((a11))

    ind_0 = tf.stack([ind0[:, 0], ind0[:, 1]], axis=1)
    ind_1 = tf.stack([ind1[:, 0], ind1[:, 1] + tf.cast(tf.shape(parent_array)[1] / 2, tf.int64)], axis=1)

    # Here it seems that we recombine not the two closest ones in the parent array, but we recombine sample
    # i with sample i + num_parent_samples/2, so in the case of "manual recombination selection", or a weighted
    # probability one, just add them subsequently onto the end of the parent array at all times

    parent_0_vals = tf.gather_nd(parent_array, ind_0)
    parent_1_vals = tf.gather_nd(parent_array, ind_1)

    st0 = tf.scatter_nd(ind0, parent_0_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
    #tf.print(st0)
    st1 = tf.scatter_nd(ind1, parent_1_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
    #tf.print(st1)

    offspring = st0 + st1

    return offspring

def create_offspring_unlooped_CN(parent_array, n_markers,gamma = 0.1):
    # Using this function we create all offspring at once, instead of creating them one at a time as in a previous implementation
    # TODO: Write clearer comments on what I am doing here, now this is pretty raw
    bp_per_cm = 1000000.
    cm_dist = tf.linspace(1., 100e6, n_markers) / bp_per_cm / 100.

    # bim_file = "/home/x_fitho/ContrastiveLosses/ContrastiveLosses_gcae/gcae/Data/dog/dog_filtered.bim"

    # snp_data = np.genfromtxt(bim_file, usecols=(0, 1, 2, 3, 4, 5), dtype=str)[0:n_markers]
    # cm_dist = snp_data[:, 3].astype(int) / bp_per_cm / 100

    recomb_prob = 1 / 2 * (1 - tf.math.exp(-4 * (cm_dist[1:] - cm_dist[:-1]))) *1
    recomb_prob = tf.cast(recomb_prob, tf.float32)
    recomb_prob2 = tf.tile(tf.reshape(recomb_prob, [n_markers - 1, 1]),
                           [1, tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)])

    u = tf.random.uniform(shape=tf.shape(recomb_prob2), minval=0, maxval=1)

    modulo = 1000
    dic_vec = tf.where(tf.random.uniform(shape  = (modulo,)) < gamma, 1,0)
    a11 =  tf.math.cumsum(tf.cast(u < recomb_prob2, tf.int32), axis=0) + tf.random.uniform([1,tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)], minval = 0, maxval = modulo, dtype = tf.int32)
    a1 = a11 % modulo

    a1 = tf.gather(dic_vec, a1)

    a = tf.concat([a1, a1[tf.newaxis, -1, :]], axis=0) # append the last value, which has been removed due to the cumsum
    ind0 = tf.where(a == 0)

    ind1 = tf.where(a == 1)


    #tf.print(tf.shape(ind1)[0] / tf.shape(ind0)[0])

    ind_0 = tf.stack([ind0[:, 0], ind0[:, 1]], axis=1)
    ind_1 = tf.stack([ind1[:, 0], ind1[:, 1] + tf.cast(tf.shape(parent_array)[1] / 2, tf.int64)], axis=1)

    tf.shape(ind_0)
    # Here it seems that we recombine not the two closest ones in the parent array, but we recombine sample
    # i with sample i + num_parent_samples/2, so in the case of "manual recombination selection", or a weighted
    # probability one, just add them subsequently onto the end of the parent array at all times

    parent_0_vals = tf.gather_nd(parent_array, ind_0)
    parent_1_vals = tf.gather_nd(parent_array, ind_1)

    st0 = tf.scatter_nd(ind0, parent_0_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
    #tf.print(st0)
    st1 = tf.scatter_nd(ind1, parent_1_vals,
                        shape=[n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
    #tf.print(st1)

    offspring = st0 + st1

    return offspring





class PositionalEncoding(tf.keras.layers.Layer):  # @save
    """Positional encoding."""

    def __init__(self, num_hidden, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough P
        self.P = tf.zeros((1, max_len, num_hidden))
        # X = tf.reshape(tf.range(max_len, dtype=tf.float32), [-1,1])  /tf.math.pow(10000, tf.range(0, num_hidden, 2, dtype=tf.float32) / num_hidden)

        # This cannot be done...
        # self.P[:, :, 0::2] = tf.math.sin(X)
        # self.P[:, :, 1::2] = tf.math.cos(X)

        ### POSITIONAL ENCODING
        X = tf.reshape(tf.range(max_len, dtype=tf.float32), [-1, 1]) / tf.math.pow(max_len, tf.range(0, num_hidden, 2,
                                                                                                     dtype=tf.float32) / num_hidden)

        updates1 = tf.reshape(tf.transpose(tf.math.cos(X)), tf.shape(X)[0] * tf.shape(X)[1]) * 0.1
        updates2 = tf.reshape(tf.transpose(tf.math.sin(X)), tf.shape(X)[0] * tf.shape(X)[1]) * 0.1

        ind1 = tf.zeros((max_len * tf.shape(X)[1], 1))
        ind2cos = tf.repeat(tf.range(0, num_hidden - 1, delta=2, dtype=tf.float32), tf.shape(X)[0])[:, tf.newaxis]
        ind2sin = tf.repeat(tf.range(1, num_hidden, delta=2, dtype=tf.float32), tf.shape(X)[0])[:, tf.newaxis]
        ind3 = tf.tile(tf.range(0, max_len, delta=1, dtype=tf.float32), [tf.shape(X)[1]])[:, tf.newaxis]
        iddcos = tf.concat([ind1, ind3, ind2cos], axis=1)
        iddsin = tf.concat([ind1, ind3, ind2sin], axis=1)

        self.P = tf.tensor_scatter_nd_update(self.P, tf.cast(iddcos, tf.int32), updates1, name=None)
        self.P = tf.tensor_scatter_nd_update(self.P, tf.cast(iddsin, tf.int32), updates2, name=None)

        self.P = tf.Variable(self.P, trainable=True)

    def call(self, X, training=True, **kwargs):
        # Here, X that comes is is shape (samples, markers, 3), where 3 is [genotype, missing mask, ms var]. Want to keep missing mask.
        mask = X[:, :, 1]
        P = tf.cast(self.P, X.dtype)

        # tf.print(tf.shape(P[:, :tf.shape(X)[1], :]))
        # tf.print(tf.shape(X[:,:,0,tf.newaxis] ))
        # tf.print(tf.shape(tf.tile(P[:, :tf.shape(X)[1], :], [tf.shape(X)[0],1,1])))
        # tf.print(tf.shape(tf.tile(X[:,:,0,tf.newaxis], [1,1, tf.shape(P)[-1]] )))

        # tf.print(mask.shape)
        # a = X[:,:,0,tf.newaxis]
        # b = P[:, :tf.shape(X)[1], :]

        a = tf.tile(P[:, :tf.shape(X)[1], :], [tf.shape(X)[0], 1, 1])
        # b = tf.tile(X[:,:,0,tf.newaxis], [1,1, tf.shape(P)[-1]] )

        b = tf.tile(tf.reshape(X[:, :, 0], [tf.shape(X)[0], tf.shape(X)[1], 1]), [1, 1, tf.shape(P)[-1]])

        penc = a + b
        mask = tf.reshape(mask, [tf.shape(X)[0], tf.shape(X)[1], 1])
        x = tf.concat([penc, mask], axis=-1)

        return x  # [:tf.shape(X)[0]//2,:,:]