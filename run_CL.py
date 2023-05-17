""" Contrastive Losses

Usage:
  run_CL.py train --data=<name> --dir=<name> [--load_path=<name>]
  run_CL.py plot --data=<name> --dir=<name> [--load_path=<name>]

Options:
  -h --help     Show this screen.


"""
from docopt import docopt
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from docopt import docopt, DocoptExit
from sklearn.decomposition import PCA
import ContrastiveLosses as CL
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
sns.set()

if __name__ == '__main__':

    try:
        arguments = docopt(__doc__, version='CL')
    except DocoptExit:
        print("Invalid command. Run 'python run_CL.py --help' for more information.")
        exit(1)

    save_dir = "./Results/"+ arguments["--dir"]+"/"
    os.makedirs(save_dir,exist_ok = True)
    # Load data
    if arguments["--data"]=="mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        dataset = "mnist"
        data_size = 28
        channels = 1
        plot_labels = ["0","1","2","3","4","5","6","7","8","9"]

    if arguments["--data"]=="fashion_mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        dataset = "fashion_mnist"
        data_size = 28
        channels = 1
        plot_labels = ["Tshirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


    elif arguments["--data"]=="cifar10":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        dataset = "cifar10"
        data_size = 32
        channels = 3
        plot_labels  = ["airplane", "automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    else:
        print("Not implemented for dataset {}".format())

    def preprocess_images(images):
      images = images.reshape((images.shape[0], data_size, data_size, channels)) / 255.
      return images # np.where(images > .5, 1.0, 0.0).astype('float32')
    if dataset=="mnist" or "fashion_mnist":
        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)
        mu_train = 0#np.mean(train_images)
        std_train = 1#np.std(train_images)
        mu_test = 0#np.mean(test_images)
        std_test = 1# np.std(test_images)
    else:
        train_images = ((train_images)/255).astype('float32')
        mu_train = np.mean(train_images)
        std_train = np.std(train_images)
        train_images = (train_images - mu_train)/std_train

        test_images =( (test_images)/255).astype('float32')
        mu_test = np.mean(test_images)
        std_test = np.std(test_images)
        test_images = (test_images - mu_test)/std_test

        train_labels = train_labels[:,0]
        test_labels = test_labels[:,0]

    N = train_images.shape[0]


    marker_list = ["A", "o", "v","^", "<",">","1","2","3","4"]
    marker_list = ["$0$", "$1$", "$2$","$3$", "$4$","$5$","$6$","$7$","$8$","$9$"]
    color_list  = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray', 'pink']

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


        def __init__(self, latent_dim):
            super(Encoder, self).__init__()
            self.latent_dim = latent_dim

            inputs = tf.keras.Input(shape=(32, 32, channels))
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding="same")(inputs)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same")(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

            x = S1_B1(x)
            x = S1_B2(x)
            x = S1_B2(x)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)

            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75)(x)

            outputs = tf.keras.layers.Dense(2)(x)
            outputs = tf.keras.layers.GaussianNoise(stddev=4.66)(outputs)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Model")

        def call(self, input):
            encoding = self.model(input)
            reg_loss = 1e-7 * tf.reduce_sum(tf.math.maximum(0., tf.square(encoding) - 1 * 40000.))

            self.add_loss(reg_loss)

            return self.model(input)


    encoder = Encoder(2)

    if arguments["train"]:
        def run_optimization(model, optimizer, loss_function, input):
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

        #loss_func = CL.Triplet
        loss_func = CL.CentroidSS
        """
        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate =   0.01, # 0.000001, #for triplet, 0.1 for scaled centroid on mnist
            first_decay_steps = 3e4,
            t_mul=1,
            m_mul=.95,
            alpha=1e-5,
            name=None
        )"""
        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate =   0.01, # 0.000001, #for triplet, 0.1 for scaled centroid on mnist
            first_decay_steps = 1e5,
            t_mul=1,
            m_mul=.95,
            alpha=1e-5,
            name=None
        )
        #optimizer = tf.optimizers.Adam(learning_rate = schedule, beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)
        optimizer = tf.optimizers.SGD(learning_rate=schedule, momentum=0.9,nesterov=True)  # , beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)

        # Load the weights of a previous run:
        if arguments["--load_path"]:
            encoder.load_weights(arguments["--load_path"])
        print(encoder.model.summary())


        if dataset =="mnist" or "fashion_mnist":
            rot = tf.keras.layers.RandomRotation(factor = 0.1, interpolation = 'bilinear', fill_mode = "constant", fill_value = 0.0, )
            shift =tf.keras.layers.RandomTranslation(0.2,0.2,fill_mode='constant',fill_value = 0)
            zoom = tf.keras.layers.RandomZoom(height_factor = [0.,0.7],width_factor=[0.,0.7],fill_mode='constant',interpolation='bilinear',seed=None,fill_value=0.0,)
            contrast = tf.keras.layers.RandomContrast(factor=0.4)
            flip = tf.keras.layers.RandomFlip(mode="horizontal")
        elif dataset == "cifar10":
            rot = tf.keras.layers.RandomRotation(factor=0.1, interpolation='bilinear')
            shift = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest', fill_value=0)
            zoom = tf.keras.layers.RandomZoom(height_factor=[-0.3, -0.1], width_factor=[-0.3, -0.1], fill_mode='constant',
                                              interpolation='bilinear', seed=None, fill_value=0.0, )
            contrast = tf.keras.layers.RandomContrast(factor=0.4)
            flip = tf.keras.layers.RandomFlip(mode="horizontal")

        epochs =    1000
        batch_size = 200
        num_samples = train_images.shape[0]
        loss = []
        times = []
        epoch_vec = []
        for e in range(epochs):
            if e%1 == 0 :
                samples_to_plot = 5000

                if dataset=="mnist" or "fashion_mnist":
                    emb = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(train_images[:samples_to_plot, :]), training = False)
                elif dataset =="cifar10":
                    emb = encoder(train_images[:samples_to_plot, :])


                plt.figure()
                ax = plt.gca()
                """
                for i in range(10):
                    inds = np.where(train_labels[:samples_to_plot] == i)[0]
                    marker = "$" + str(i) + "$"
                    ax.scatter(emb.numpy()[inds, 0], emb.numpy()[inds, 1],  marker=marker_list[i], color=color_list[i] ) # ,s = 120,  edgecolor='black',linewidth=0.00)
                """

                D = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": tf.gather(plot_labels, tf.cast(train_labels[:samples_to_plot], tf.int32)).numpy().astype(str)})
                sns.scatterplot(data=D, x="x", y="y", hue="label", palette=sns.color_palette("tab10"),
                                legend="brief",hue_order = plot_labels)  # ,# = emb.numpy()[:, 0], y = emb.numpy()[:, 1], hue = np.array(color_list)[train_labels[samples_to_plot*ii:samples_to_plot*(ii+1)]], legend='brief')

                plt.title("Epoch: {}".format(e))
                plt.legend(fontsize='x-small', title_fontsize='40')

                plt.savefig(save_dir+"Epoch: {}.pdf".format(e))
                plt.close()



            t0 = time.perf_counter()
            current_batch = 0
            a = 0
            train_images2 = tf.random.shuffle(train_images[:num_samples, :])

            while current_batch < num_samples:

                # Augmentation:

                if dataset =="mnist" or "fashion_mnist":
                    input = train_images2[current_batch:current_batch+batch_size,:]
                    input = shift(rot(zoom(tf.concat([input,input], axis = 0))))

                elif dataset =="cifar10":
                    input = train_images2[current_batch:current_batch + batch_size, :]
                    #input = tf.concat([input,input], axis = 0)
                    input = tf.concat([tf.image.random_hue(flip((input)), 0.05), tf.image.random_hue(flip((input)), 0.05)],axis=0)

                    input = rot(shift(zoom(input)))

                    #noise = tf.math.abs(tf.random.normal(tf.shape(input), mean = 0, stddev =0.3 ))
                    input = input  #+ tf.cast((noise>0.5),tf.float32)* tf.clip_by_value(noise,1,1)



                    input = tf.clip_by_value(input, tf.reduce_min(input),tf.reduce_max(input))

                if current_batch%(batch_size*10) == 0: # Save image of augmented samples and where they get mapped, used in development - checking augmentations.
                    plt.figure()
                    plt.subplot(223)
                    plt.imshow(input[0]*std_train + mu_train, cmap="gray")
                    plt.subplot(224)
                    plt.imshow(input[batch_size]*std_train + mu_train, cmap="gray")
                    if dataset=="mnist" or "fashion_mnist":
                        emb2 = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(input))
                    elif dataset == "cifar10":
                        emb2 = encoder(input)

                    plt.subplot(211)
                    plt.scatter(emb2.numpy()[:, 0], emb2.numpy()[:, 1], color = 'k')
                    plt.scatter(emb2.numpy()[0, 0], emb2.numpy()[0, 1],color ='g')
                    plt.scatter(emb2.numpy()[batch_size, 0], emb2.numpy()[batch_size, 1],color = 'b')

                    distances = tf.sqrt(
                        tf.reduce_sum((emb2[:, :, tf.newaxis] - tf.transpose(emb2[:, :, tf.newaxis])) ** 2, axis=1))

                    plt.savefig(save_dir+"augmentation.pdf")

                    plt.close()
                if dataset == "mnist" or "fashion_mnist":
                    input = tf.keras.layers.ZeroPadding2D(padding=2)(input)
                elif dataset == "cifar10":

                    input = input

                a += run_optimization(encoder, optimizer, loss_func, input)
                current_batch +=batch_size


            print("Epoch {}, loss:  {}, learning rate: {} ".format(e, a/num_samples, optimizer._decayed_lr(var_dtype=tf.float32).numpy()))

            if e % 100 ==0:
                print("saving model at epoch {}".format(e))
                encoder.save_weights(save_dir+'saved_model/epoch_{}'.format(e))

            loss.append(a)
            times.append(time.perf_counter()-t0)
            epoch_vec.append(e)

            plt.figure()
            plt.plot(epoch_vec, loss)
            plt.savefig(save_dir+"loss.pdf")
            plt.close()

    elif arguments["plot"]:

        epochs = []
        for file in os.listdir(save_dir+"saved_model"):
            if file[-5:] == "index":
                epochs.append(file[:-6])

        for epoch in np.sort(epochs):

            encoder.load_weights(save_dir+"saved_model/"+epoch)

            batch_size = 1000
            num_samples = train_images.shape[0]
            # samples_to_plot*i:samples_to_plot*(i+1)
            num_batches = num_samples // batch_size
            full_emb = np.empty((0, 2))

            """ This plots with the numbers as markers, can get pretty messy.
            plt.figure()
            for ii in range(num_batches):
                emb = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(train_images[samples_to_plot*ii:samples_to_plot*(ii+1), :]), training=False)
                full_emb = np.append(full_emb, emb, axis = 0)
                ax = plt.gca()
                for i in range(10):
                    inds = np.where(train_labels[samples_to_plot*ii:samples_to_plot*(ii+1)] == i)[0]
                    marker = "$" + str(i) + "$"
                    ax.scatter(emb.numpy()[inds, 0], emb.numpy()[inds, 1], marker=marker_list[i],
                               color=color_list[i])#, s = 120,  edgecolor='black',linewidth=0.02)
            plt.title("CentroidSS on MNIST:")
            plt.savefig("mnist_example.pdf")
            plt.close()
            plt.figure()
    
            """

            plot_train = True
            if plot_train:
                images = train_images
                labels = train_labels
            else:
                images = test_images
                labels = test_labels

            full_emb = np.empty((0, 2))
            for ii in range(num_batches):
                if dataset=="mnist" or "fashion_mnist":
                    emb = encoder(tf.keras.layers.ZeroPadding2D(padding=2)(images[batch_size * ii:batch_size * (ii + 1), :]),
                              training=False)
                elif dataset =="cifar10":
                    emb = encoder(images[batch_size * ii:batch_size * (ii + 1), :], training=False)
                full_emb = np.append(full_emb, emb, axis=0)

            """
            D = pd.DataFrame({"x": full_emb[:, 0], "y": full_emb[:, 1], "color": labels})
            sns.scatterplot(data=D, x="x", y="y", hue="color", palette=sns.color_palette("tab10"),
                            legend="brief")  # ,# = emb.numpy()[:, 0], y = emb.numpy()[:, 1], hue = np.array(color_list)[train_labels[samples_to_plot*ii:samples_to_plot*(ii+1)]], legend='brief')
            """
            D = pd.DataFrame({"x": full_emb[:, 0], "y": full_emb[:, 1], "label": tf.gather(plot_labels,
                                                                                 tf.cast(labels,
                                                                                         tf.int32)).numpy().astype(
                str)})
            sns.scatterplot(data=D, x="x", y="y", hue="label", palette=sns.color_palette("tab10"),
                            legend="brief",
                            hue_order=plot_labels)  # ,# = emb.numpy()[:, 0], y = emb.numpy()[:, 1], hue = np.array(color_list)[train_labels[samples_to_plot*ii:samples_to_plot*(ii+1)]], legend='brief')

            plt.legend(fontsize='x-small', title_fontsize='40')

            plt.title("Contrastive Learning on " +dataset+ " "  +epoch)
            plt.savefig(save_dir+dataset +"_"+epoch+".png")
            plt.close()

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(full_emb, labels)
        score = neigh.score(full_emb, labels)

        print("3 Nearest neighbour classification score: {}".format(score))

        pca = PCA(n_components=2)
        pca.fit(np.reshape(train_images, [num_samples, data_size**2*channels])[:num_samples, :])
        X_PCA = pca.transform(np.reshape(train_images, [num_samples, data_size**2*channels])[:num_samples, :])
        plt.figure()

        D = pd.DataFrame({"x": X_PCA[:, 0], "y": X_PCA[:, 1], "color": train_labels})
        sns.scatterplot(data=D, x="x", y="y", hue="color", palette=sns.color_palette("tab10"), legend="brief")
        plt.savefig(save_dir+"pca.pdf")

        neigh3 = KNeighborsClassifier(n_neighbors=3)
        neigh3.fit(X_PCA[:, 0:1], train_labels[:N])
        scorePCA = neigh3.score(X_PCA[:, 0:1], train_labels[:N])
        print(" PCA classification score: {}".format(scorePCA))

        N = 60000
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=3).fit_transform(np.reshape(train_images, [num_samples, data_size**2*channels])[:N, :])

        neigh2 = KNeighborsClassifier(n_neighbors=3)
        neigh2.fit(X_embedded[:, 0:1], train_labels[:N])
        scoretsne = neigh2.score(X_embedded[:, 0:1], train_labels[:N])
        print(" t-SNE classification score: {}".format(scoretsne))

        plt.figure()
        D = pd.DataFrame({"x": X_embedded[:, 0], "y": X_embedded[:, 1], "color": train_labels})
        sns.scatterplot(data=D, x="x", y="y", hue="color", palette=sns.color_palette("tab10"), legend="brief")
        plt.savefig(save_dir+"tsne.pdf")
