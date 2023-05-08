
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
import ContrastiveLosses as CL
import time

sns.set()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)


train_images
N = 6000


X_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=3).fit_transform(np.reshape(train_images,[60000,784])[:N,:])

marker_list = ["A", "o", "v","^", "<",">","1","2","3","4"]
marker_list = ["$0$", "$1$", "$2$","$3$", "$4$","$5$","$6$","$7$","$8$","$9$"]


color_list  = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray', 'pink']

mlist = [marker_list[i] for i in train_labels ]


markers = ["$x$","$y$","$z$"]

plt.figure()
ax = plt.gca()
for i in range(10):
    inds = np.where(train_labels[:N] == i)
    marker = "$"+str(i) +"$"
    ax.plot(X_embedded[inds,0], X_embedded[inds,1],marker =marker_list[i], color = color_list[i])
   # ax.plot((i+1)*[i,i+1],marker=marker_list[i],lw=0)

plt.savefig("tsne.pdf")


pca = PCA(n_components=2)
pca.fit(np.reshape(train_images,[60000,784])[:60000,:])
X_PCA = pca.transform(np.reshape(train_images,[60000,784])[:60000,:])
plt.figure()
ax = plt.gca()
for i in range(10):
    inds = np.where(train_labels[:N] == i)
    marker = "$"+str(i) +"$"
    ax.plot(X_PCA[inds,0], X_PCA[inds,1],marker =marker_list[i], color = color_list[i])

plt.savefig("pca.pdf")

def S1_B2(x):
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding="same")(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)

    y = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="same")(y)
    y = tf.keras.layers.BatchNormalization()(y)


    x = tf.keras.layers.Add()([x, y])
    return tf.keras.layers.Activation("relu")(x)

def S1_B1(x):
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding="same")(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)

    y = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="same")(y)
    y = tf.keras.layers.BatchNormalization()(y)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, y])
    return tf.keras.layers.Activation("relu")(x)
class Encoder(tf.keras.Model):
    """Convolutional autoencoder."""

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim


        inputs =  tf.keras.Input(shape=(32, 32, 1))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding ="same")(inputs)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding ="same")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding ="same")(x)
        x = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(x)

        x = S1_B1(x)
        x = S1_B2(x)
        x = S1_B2(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(75,activation = "relu")(x)
        x = tf.keras.layers.Dense(75, activation="relu")(x)
        x = tf.keras.layers.Dense(75, activation="relu")(x)

        x = tf.keras.layers.Dense(75,activation = "relu")(x)
        x = tf.keras.layers.Dense(75)(x)

        outputs  = tf.keras.layers.Dense(2)(x)
        outputs = tf.keras.layers.GaussianNoise(stddev = 4.66 ) (outputs)
        self.model = tf.keras.Model(inputs=inputs, outputs = outputs, name = "Model")


    def call(self, input):
        encoding = self.model(input)
        reg_loss = 1e-8 * tf.reduce_sum(tf.math.maximum(0., tf.square(encoding) - 1 * 40000.))

        self.add_loss(reg_loss)

        return self.model(input)

def run_optimization(model, optimizer, loss_function, input ):
    '''
    Run one step of optimization process based on the given data.


    :param model: a tf.keras.Model
    :param optimizer: a tf.keras.optimizers
    :param loss_function: a loss function
    :param input: input data
    :param targets: target data
    :return: value of the loss function
    '''

    with tf.GradientTape() as g:
        output = model(input, training= True)
        loss_value = loss_function(anchors = output[:tf.shape(output)[0] // 2, :], positives = output[tf.shape(output)[0] // 2:, :])
        loss_value += tf.nn.scale_regularization_loss(tf.reduce_sum(model.losses))
    gradients = g.gradient(loss_value, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value



loss_func = CL.Triplet

schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate =  0.000001, #for triplet, 0.1 for scaled centroid
    first_decay_steps = 3e4,
    t_mul=1,
    m_mul=.95,
    alpha=1e-5,
    name=None
)
optimizer = tf.optimizers.Adam(learning_rate = schedule, beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)
encoder = Encoder(2)
optimizer = tf.optimizers.SGD(learning_rate=schedule, momentum=0.9,nesterov=True)  # , beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)



# Load the weights of a previous run:

#encoder.load_weights('./Examples/triplet/saved_model/epoch_900')
print(encoder.model.summary())



rot = tf.keras.layers.RandomRotation(factor = 0.1, interpolation = 'bilinear')
shift =tf.keras.layers.RandomTranslation(0.2,0.2,fill_mode='constant',fill_value = 0)
zoom = tf.keras.layers.RandomZoom(height_factor = [0.,0.7],width_factor=[0.,0.7],fill_mode='constant',interpolation='bilinear',seed=None,fill_value=0.0,)

epochs =    1000
batch_size = 200
num_samples = 60000
loss = []
times = []
epoch_vec = []


for e in range(epochs):

    if e%1 == 0 :
        samples_to_plot = 5000

        emb = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(train_images[:samples_to_plot, :]), training = False)
        #emb = encoder(train_images[:samples_to_plot, :])
        plt.figure()
        ax = plt.gca()
        for i in range(10):
            inds = np.where(train_labels[:samples_to_plot] == i)[0]
            marker = "$" + str(i) + "$"
            ax.scatter(emb.numpy()[inds, 0], emb.numpy()[inds, 1],  marker=marker_list[i], color=color_list[i] ) # ,s = 120,  edgecolor='black',linewidth=0.00)
        plt.title("Epoch: {}".format(e))
        plt.savefig("Epoch123: {}.pdf".format(e))
        plt.close()
    t0 = time.perf_counter()
    current_batch = 0
    a = 0
    train_images2 = tf.random.shuffle(train_images[:num_samples, :])

    while current_batch < num_samples:

        input = train_images2[current_batch:current_batch+batch_size,:]
        input = tf.concat([input,input], axis = 0)
        input = shift(rot(zoom(input)))


        if current_batch%(batch_size*10) == 0:
            plt.figure()
            plt.subplot(223)
            plt.imshow(input[0], cmap="gray")
            plt.subplot(224)
            plt.imshow(input[batch_size], cmap="gray")

            #emb2 = encoder(input)
            emb2 = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(input))

            plt.subplot(211)
            plt.scatter(emb2.numpy()[:, 0], emb2.numpy()[:, 1], color = 'k')
            plt.scatter(emb2.numpy()[0, 0], emb2.numpy()[0, 1],color ='g')
            plt.scatter(emb2.numpy()[batch_size, 0], emb2.numpy()[batch_size, 1],color = 'b')

            distances = tf.sqrt(tf.reduce_sum((emb2[:, :, tf.newaxis] - tf.transpose(emb2[:, :, tf.newaxis])) ** 2, axis=1))

            plt.savefig("test.pdf")
            plt.close()
        input = tf.keras.layers.ZeroPadding2D(padding=2)(input)

        a += run_optimization(encoder, optimizer, loss_func, input)
        current_batch +=batch_size
    print("Epoch {}, loss:  {}, learning rate: {} ".format(e, a/num_samples, optimizer._decayed_lr(var_dtype=tf.float32).numpy()))

    if e % 100 ==0:
        print("saving model at epoch {}".format(e))
        encoder.save_weights('./saved_model/epoch_{}'.format(e))

    loss.append(a)
    times.append(time.perf_counter()-t0)
    epoch_vec.append(e)

    plt.figure()
    plt.plot(epoch_vec, loss)
    plt.savefig("loss.pdf")
    plt.close()


