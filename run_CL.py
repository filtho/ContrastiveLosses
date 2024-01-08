""" Contrastive Losses

Usage:
  run_CL.py train --data=<name> --dir=<name> [--load_path=<name> --config=<name>]
  run_CL.py plot --data=<name> --dir=<name> [--load_path=<name>]

Options:
  -h --help     Show this screen.


"""
import os
import time
from docopt import docopt, DocoptExit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

if "SLURM_NTASKS_PER_NODE" in os.environ:
	if int(os.environ["SLURM_NTASKS_PER_NODE"]) > 1:
		if int(os.environ["SLURM_NTASKS_PER_NODE"]) != len((os.environ["CUDA_VISIBLE_DEVICES"]).split(",")):
			print("Need to have either just 1 process for single-node-multi GPU jobs, or the same number of processes as gpus.")
			exit(3)
		else:
			#If we use more than one task, we need to set devices. For more than 1 process on one node, it will otherwise try to use all gpus on all processes
			os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]


from datetime import datetime
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition  import PCA
from sklearn.cluster import KMeans
import ContrastiveLosses as CL
from set_tf_config_berzelius import set_tf_config
import json
from keras import backend as K
sns.set()


def chief_print(str):

    if "isChief" in os.environ:

        if os.environ["isChief"] == "true":
            tf.print(str)
    else:
        tf.print(str)

def _isChief():

	if "isChief" in os.environ:

		if os.environ["isChief"] == "true":
			return True
		else:
			return False
	else:
		return True

def write_to_csv(destination, values, epoch_number):
    """
    Appends the value in values and the current epoch to the csv file specified in destination.
    """
    if _isChief():
        values_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(epoch_number,tf.float32)), [1, 1]), tf.reshape(tf.cast(values, tf.float32),[1,tf.shape(values)[0]])],axis = 1).numpy()[0]

        with open(destination, 'a', newline='',  encoding="utf-8") as fd:
            writer = csv.writer(fd)
            writer.writerow([str(i) for i in values_write])

def compute_KNN_accuracy(data, labels, k = 3):
    """
    Computes the knn-classification accuracy. 
    Assuming that data contains the coordinates for the samples, in the format [n_samples, dimension]
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, labels)
    score = knn.score(data,labels)

    return score

def k_means_clustering(data, n, labels, plot_labels = [], e = [], plot = True):
    """
   
    Assuming that data contains the coordinates for the samples, in the format [n_samples, dimension]
    """

    kmeans = KMeans(n_clusters = n, n_init = "auto").fit(data)
    size = tf.shape(data)[0].numpy()
    labels_to_plot = tf.zeros((size,),dtype = tf.int32)

    for i in range(10):
        inds = tf.where(kmeans.labels_ == i)
        vals = tf.squeeze(tf.gather(labels, inds))
        x, _, count = tf.unique_with_counts(vals)
        labels_to_plot+=tf.scatter_nd(inds, tf.ones_like(vals,dtype = tf.int32) * tf.cast(x[tf.argmax(count)],tf.int32), shape =tf.constant([size],dtype= tf.int64) )


    acc = tf.reduce_sum(tf.cast(tf.squeeze(labels_to_plot) == tf.squeeze(tf.cast(labels,tf.int32)), tf.int32))/size
    if plot:

        # below is taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
        h = 0.02 
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = tf.reduce_min(data[:, 0]) - 1, tf.reduce_max(data[:, 0])+ 1
        y_min, y_max = tf.reduce_min(data[:, 1]) - 1, tf.reduce_max(data[:, 1]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        plt.figure()

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
        )



        D = pd.DataFrame({"x": data[:, 0], "y": data[:, 1], "label": tf.gather(plot_labels, tf.cast(labels_to_plot,tf.int32)).numpy().astype(str)})
        sns.scatterplot(data=D, x="x", y="y", hue="label", palette=sns.color_palette("tab10"), legend="brief",hue_order = plot_labels)  
        plt.title("Epoch: {}".format(e))
        plt.legend(fontsize='x-small', title_fontsize='40')
        plt.savefig(save_dir+"k-means_clustering_epoch: {}.pdf".format(e))
        plt.close()

    return acc




if __name__ == '__main__':
    try:
        arguments = docopt(__doc__, version='CL')
    except DocoptExit:
        chief_print("Invalid command. Run 'python run_CL.py --help' for more information.")
        exit(1)

    if "SLURMD_NODENAME" in os.environ:

        slurm_job = 1
        addresses, chief, num_workers = set_tf_config()
        isChief = os.environ["SLURMD_NODENAME"] == chief
        os.environ["isChief"] = json.dumps(str(isChief))
        chief_print(num_workers)

        if num_workers > 1  and  not arguments["plot"]:

            strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver(),
                                                                communication_options=tf.distribute.experimental.CommunicationOptions(
                                                                implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
                                                                )
        

            if  not isChief:
                tf.get_logger().setLevel('ERROR')
                #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        
            num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))

            chief_print(tf.config.list_physical_devices(device_type='GPU'))
            gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
            #chief_print(gpus)


        else:
            if not isChief:
                print("Work has ended for this worker, now relying only on the Chief.")
                exit(0)
            tf.print(tf.config.list_physical_devices(device_type='GPU'))
            tf.print(tf.test.gpu_device_name())
            num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))

            chief_print(tf.config.list_physical_devices(device_type='GPU'))
            gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
            chief_print(gpus)
            strategy =  tf.distribute.MirroredStrategy(devices = gpus, cross_device_ops=tf.distribute.NcclAllReduce())

            slurm_job = 0
        os.environ["isChief"] = json.dumps((isChief))

    else:
        isChief = True
        slurm_job = 0
        num_workers = 1

        strategy =  tf.distribute.MirroredStrategy()

    num_devices = strategy.num_replicas_in_sync

    save_dir = "./Results/"+ arguments["--dir"]+"/"
    os.makedirs(save_dir,exist_ok = True)
    os.makedirs(save_dir+"stats", exist_ok=True)
    # Load data
    if arguments["--data"]=="mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        dataset = "mnist"
        data_size = 28
        channels = 1
        plot_labels = ["0","1","2","3","4","5","6","7","8","9"]

    elif arguments["--data"]=="fashion_mnist":
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
        chief_print(f"Not implemented for dataset {arguments['--data']}")


    def preprocess_images(img):

        """ Basic preprocessing of MNIST images. """

        images_processed = img.reshape((img.shape[0], data_size, data_size, channels)) / 255.
        return images_processed # np.where(images > .5, 1.0, 0.0).astype('float32')


    if dataset=="mnist" or dataset=="fashion_mnist":
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

    class concat_layer(tf.keras.layers.Layer):

        def call(self,inputs, training = True):
            if training:
                return tf.keras.layers.Concatenate(axis = 0)([inputs, inputs])
            else:
                return inputs


    class SnP(tf.keras.layers.Layer):
        """
        Salt and pepper noise, adapted from https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras """
        def call(self,inputs, training = True):
            
            def noised():
                shp = K.shape(inputs)[1:]
                mask_select = K.random_bernoulli(shape=shp, p=0.1)
                mask_noise = K.random_bernoulli(shape=shp, p=0.5) # salt and pepper have the same chance
                out = inputs * (1-mask_select) + mask_noise * mask_select
                return out

            return K.in_train_phase(noised(), inputs, training=training)
    
    class Encoder(tf.keras.Model):
        """ A class defining the model architecture, and the call logic."""

        def __init__(self, latent_dim):
            super(Encoder, self).__init__()
            self.latent_dim = latent_dim

            inputs = tf.keras.Input(shape=(data_size, data_size, channels))

            x = self.augmentation(inputs)

            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding="same")(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same")(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

            x = self.S1_B1(x)
            x = self.S1_B2(x)
            x = self.S1_B2(x)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75, activation="relu")(x)
            x = tf.keras.layers.Dense(75)(x)

            outputs = tf.keras.layers.Dense(2)(x)
            outputs = tf.keras.layers.GaussianNoise(stddev=  4.66 )(outputs) # 4.66 , 2/ 9 used for all experiments in report
            self.model = tf.keras.Model(inputs=inputs, outputs=(outputs, x), name="Model")

        def call(self, inputs, repr = False):
            encoding, x = self.model(inputs)
            
            
            reg_loss = 200 * 1e-1 * tf.reduce_sum(tf.math.maximum(0., tf.square(encoding) - 1 * 40000.) / 40000.) / tf.cast(tf.shape(encoding)[0], tf.float32)
            #reg_loss = 1e-1 * tf.reduce_sum(tf.math.maximum(0., tf.square(encoding[:,0]) - 1 * 250000.)/250000.+ tf.math.maximum(0., tf.square(encoding[:,1]) -  40000.) / 40000.) / tf.cast(tf.shape(encoding)[0], tf.float32)



            self.add_loss(reg_loss)


            if repr:
                return encoding, x
            else:
                return encoding

        def augmentation(self,inputs):


            if arguments["--data"]=="mnist" :
                rot = tf.keras.layers.RandomRotation(factor = 0.1, interpolation = 'bilinear', fill_mode = "constant", fill_value = 0.0)
                shift =tf.keras.layers.RandomTranslation(0.2,0.2,fill_mode='constant',fill_value = 0)
                zoom = tf.keras.layers.RandomZoom(height_factor = [0.,0.7],width_factor=[0.,0.7],fill_mode='constant',interpolation='bilinear',seed=None,fill_value=0.0, )

                inputs2 = concat_layer()(inputs)
                x = SnP()(shift(rot(zoom(inputs2))))

            elif arguments["--data"]=="fashion_mnist":
                rot = tf.keras.layers.RandomRotation(factor = 0.1, interpolation = 'bilinear', fill_mode = "constant", fill_value = 0.0)
                shift =tf.keras.layers.RandomTranslation(0.2,0.2,fill_mode='constant',fill_value = 0)
                zoom = tf.keras.layers.RandomZoom(height_factor = [0.,0.7],width_factor=[0.,0.7],fill_mode='constant',interpolation='bilinear',seed=None,fill_value=0.0, )
                flip = tf.keras.layers.RandomFlip(mode="horizontal")

                inputs2 = concat_layer()(inputs)
                x = shift(rot(zoom(flip(inputs2))))

            elif arguments["--data"]=="cifar10":
                rot = tf.keras.layers.RandomRotation(factor=0.1, interpolation='bilinear')
                shift = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest', fill_value=0)
                zoom = tf.keras.layers.RandomZoom(height_factor=[-0.3, -0.1], width_factor=[-0.3, -0.1], fill_mode='constant',
                                                interpolation='bilinear', seed=None, fill_value=0.0, )
                contrast = tf.keras.layers.RandomContrast(factor=0.4)
                flip = tf.keras.layers.RandomFlip(mode="horizontal")

                x = rot(flip(concat_layer()(inputs)))

            return x

        def S1_B2(self,x):
            """ A short implementation for a residual block"""
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

        def S1_B1(self,x):
            """ A short implementation for a "introductory" residual block """
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



    with strategy.scope():
        encoder = Encoder(2)

    if arguments["train"]:

        with strategy.scope():
            @tf.function
            def run_optimization(model, opt, loss_function, inputs):
                '''
                Run one step of optimization process based on the given data.


                :param model: a tf.keras.Model
                :param opt: a tf.keras.optimizers
                :param loss_function: a loss function
                :param inputs: input data
                :return: value of the loss function
                '''

                with tf.GradientTape() as g:
                    output = model(inputs, training= True)
                    loss_value = loss_function(anchors = output[:tf.shape(output)[0] // 2, :], positives = output[tf.shape(output)[0] // 2:, :], indices = None)
                    loss_value += tf.nn.scale_regularization_loss(tf.reduce_sum(model.losses))
                gradients = g.gradient(loss_value, model.trainable_variables)

                opt.apply_gradients(zip(gradients, model.trainable_variables))
                return loss_value

            @tf.function
            def distributed_train_step(model, opt, loss_function, inputs):

                per_replica_losses = strategy.run(run_optimization, args=(model, opt, loss_function, inputs))

                agg_loss = strategy.reduce("SUM", per_replica_losses, axis=None)
                
                return agg_loss

            if arguments["--config"]:

                with open(arguments["--config"]) as config_file:
                    config = json.load(config_file)
		                        
                loss_def = config["loss"]

                loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
                if "args" in loss_def.keys():
                    loss_args = loss_def["args"]
                else:
                    loss_args = dict()
                loss_func = loss_class(**loss_args)

                    
                epochs =    config["epochs"]
                local_batch_size =  config["batch_size"]
                batch_size = local_batch_size * num_devices
                save_interval =  config["save_interval"]
                learning_rate = config["learning_rate"]
            else:
                loss_func = CL.centroid(n_pairs = 20,mode = 'distance_weighted_random', distance = "L2")
                epochs =    100
                local_batch_size = 200
                batch_size = local_batch_size * num_devices
                learning_rate = 0.0001    #0.000001, #for triplet, 0.1 for scaled centroid on mnist
                save_interval =  100

            schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate =    learning_rate,
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
            if _isChief():
                encoder.model.summary()


        num_samples =  train_images.shape[0]

        ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        ds = ds.shuffle(train_images.shape[0], reshuffle_each_iteration = True)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        dds  = strategy.experimental_distribute_dataset(ds)
        
        ds_validation = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        ds_validation = ds_validation.shuffle(test_images.shape[0], reshuffle_each_iteration = True)
        ds_validation = ds_validation.batch(batch_size)
        ds_validation = ds_validation.prefetch(tf.data.AUTOTUNE)

        for e in range(epochs):
            t = time.perf_counter()
           

            if "SLURM_PROCID" in os.environ:
                suffix = str(os.environ["SLURM_PROCID"])
            else:
                suffix = ""

            logs = save_dir+ "/logdir/"  + datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+ suffix
            profile = False
            if profile and e ==1: tf.profiler.experimental.start(logs)
            
            
            if e%10 == 0 :
                
                validation_embedding = None
                validation_labels = None
                validation_higher_dim= None

                for valid_image, valid_label in ds_validation:

                    emb, higher_dim = encoder(valid_image, training = False, repr = True)
                    if validation_embedding == None:
                        validation_embedding = emb
                        validation_labels = valid_label
                        validation_higher_dim = higher_dim

                    else:
                                
                        validation_embedding = tf.concat([validation_embedding, emb], axis = 0)
                        validation_labels = tf.concat([validation_labels, valid_label], axis = 0)
                        validation_higher_dim = tf.concat([validation_higher_dim, higher_dim], axis = 0)

                acc = compute_KNN_accuracy(validation_embedding, validation_labels)
                acc2 = compute_KNN_accuracy(validation_higher_dim, validation_labels)
                acc_5nn = compute_KNN_accuracy(validation_embedding, validation_labels,k =  5)
                acc_10nn = compute_KNN_accuracy(validation_embedding, validation_labels, k = 10)

                write_to_csv(save_dir+"/stats/KNN_acc.csv",[acc,acc2,acc_5nn,acc_10nn],e)

                if  _isChief():

                    plt.figure()
                    D = pd.DataFrame({"x": validation_embedding[:, 0], "y": validation_embedding[:, 1], "label": tf.gather(plot_labels, tf.cast(validation_labels,tf.int32)).numpy().astype(str)})
                    sns.scatterplot(data=D, x="x", y="y", hue="label", palette=sns.color_palette("tab10"), legend="brief",hue_order = plot_labels)  
                    plt.title("Epoch: {}".format(e))
                    plt.legend(fontsize='x-small', title_fontsize='40')
                    plt.savefig(save_dir+"Epoch: {}.pdf".format(e))
                    plt.close()
                    plt.figure()
                    acc_plot =pd.read_csv(save_dir + "/stats/KNN_acc.csv",header = None).to_numpy()
                    plt.plot(acc_plot[:,0], acc_plot[:,1], label = "2-D embedding")
                    plt.plot(acc_plot[:,0], acc_plot[:,2], label = "High-D representation")
                    plt.legend()
                    plt.savefig(save_dir + "/stats/KNN_acc.pdf")
                    plt.close()


                chief_print(f" KNN classification accuracy on the validation dataset: {acc}, on higher dim : {acc2}")

                
            t0 = time.perf_counter()
            current_batch = 0
            a = 0


            for input_data, input_label in dds:
                current_batch += batch_size

                if _isChief():
                    # This can only be done on a single gpu run as of now. It dislikes transforming stuff that are already sdistributed
                    if current_batch%(batch_size*100000) == 0: # Save image of augmented samples and where they get mapped, used in development - checking augmentations.
                        augmented_images = encoder.augmentation(input_data)

                        emb2 = encoder(augmented_images)
                        plt.figure()
                        plt.subplot(223)
                        plt.imshow(augmented_images[0]*std_train + mu_train, cmap="gray")
                        plt.subplot(224)
                        plt.imshow(augmented_images[local_batch_size]*std_train + mu_train, cmap="gray")

                    
                        plt.subplot(211)
                        plt.scatter(emb2.numpy()[:, 0], emb2.numpy()[:, 1], color = 'k')
                        plt.scatter(emb2.numpy()[0, 0], emb2.numpy()[0, 1],color ='g')
                        plt.scatter(emb2.numpy()[local_batch_size, 0], emb2.numpy()[local_batch_size, 1],color = 'b')

                        distances = tf.sqrt(
                            tf.reduce_sum((emb2[:, :, tf.newaxis] - tf.transpose(emb2[:, :, tf.newaxis])) ** 2, axis=1))
                        #TODO: show which samples have been used as negatives :)


                        plt.savefig(save_dir+"augmentation.pdf")
                        plt.close()
                a += distributed_train_step(encoder, optimizer, loss_func, input_data)
               
           
            #chief_print("Epoch {}, loss:  {}, learning rate: {} time: {}".format(e, a/num_samples,optimizer._current_learning_rate.numpy() , time.perf_counter() - t ))# optimizer._decayed_lr(var_dtype=tf.float32).numpy()))
            t1 = time.perf_counter()-t0
            chief_print(f"Epoch {e}, Loss: {a/num_samples}, time: {t1}") # , loss:  {}, learning rate: {} time: {}".format(e, a/num_samples,optimizer._current_learning_rate.numpy() , time.perf_counter() - t ))# optimizer._decayed_lr(var_dtype=tf.float32).numpy()))


            if _isChief():
                write_to_csv(save_dir+"stats/loss.csv", [a], e)
                write_to_csv(save_dir+"stats/time.csv", [t1], e)

                weights_file_prefix = save_dir+'saved_model/epoch_{}'.format(e)
                plt.figure()
                loss_plot =pd.read_csv(save_dir + "/stats/loss.csv",header = None).to_numpy()
                plt.plot(loss_plot[:,0], loss_plot[:,1])
                plt.savefig(save_dir + "/stats/loss.pdf")
                plt.close()
  
            else:
                weights_file_prefix ="/scratch/local/"+ str(e)+os.environ["SLURM_PROCID"] # Save to some junk directory, /scratch/local on Berra is a temp directory that deletes files after job is done.
            
            if e % save_interval ==0:
                chief_print("saving model at epoch {}".format(e))
                encoder.save_weights(weights_file_prefix)
                k_means_clustering(validation_embedding,n = 10, labels = validation_labels, plot_labels=plot_labels, e = e)


           
        if profile: tf.profiler.experimental.stop()



    
    elif arguments["plot"]:

        epochs = []
        for file in os.listdir(save_dir+"saved_model"):
            if file[-5:] == "index":
                epochs.append(file[:-6])

        for epoch in np.sort(epochs):

            encoder.load_weights(save_dir+"saved_model/"+epoch)

            batch_size = 1000
            num_samples = train_images.shape[0]
            num_batches = num_samples // batch_size
            full_emb = np.empty((0, 2))

            plot_train = True
            if plot_train:
                images = train_images
                labels = train_labels
            else:
                images = test_images
                labels = test_labels

            full_emb = np.empty((0, 2))
            for ii in range(num_batches):

                emb = encoder(images[batch_size * ii:batch_size * (ii + 1), :], training=False)
                full_emb = np.append(full_emb, emb, axis=0)



            #plt.figure(figsize=(10,5))
            plt.figure()
            D = pd.DataFrame({"x": full_emb[:, 0], "y": full_emb[:, 1],
             "label": tf.gather(plot_labels,tf.cast(labels,tf.int32)).numpy().astype(str)})
            sns.scatterplot(data=D, x="x", y="y", hue="label", palette=sns.color_palette("tab10"),
                            legend="brief",
                            hue_order=plot_labels)  # ,# = emb.numpy()[:, 0], y = emb.numpy()[:, 1], hue = np.array(color_list)[train_labels[samples_to_plot*ii:samples_to_plot*(ii+1)]], legend='brief')

            plt.legend(fontsize='x-small', title_fontsize='40')
            score = compute_KNN_accuracy(full_emb, labels)
            #plt.axis("equal")
            plt.title(f"Contrastive Learning on {dataset} {epoch } KNN classification accuracy: {score}")
            plt.savefig(save_dir+dataset +"_"+epoch+".png")
            plt.close()
            kmeans_acc = k_means_clustering(full_emb,n = 10, labels = labels, plot_labels=plot_labels, e = epoch, plot = True)
            
            tf.print((f" KNN classification accuracy : {score}"))
            tf.print((f" KMeans classification accuracy: : {kmeans_acc}"))

        chief_print("3 Nearest neighbour classification score: {}".format(score))

        pca = PCA(n_components=2)
        pca.fit(np.reshape(train_images, [num_samples, data_size**2*channels])[:num_samples, :])
        X_PCA = pca.transform(np.reshape(train_images, [num_samples, data_size**2*channels])[:num_samples, :])
        plt.figure()
        scorePCA = compute_KNN_accuracy(X_PCA[:, 0:2], train_labels[:N])

        D = pd.DataFrame({"x": X_PCA[:, 0], "y": X_PCA[:, 1], "color": train_labels})
        sns.scatterplot(data=D, x="x", y="y", hue="color", palette=sns.color_palette("tab10"), legend="brief")
        plt.title(f"PCA, KNN classification accuracy: {scorePCA}")
        plt.savefig(save_dir+"pca.pdf")


        chief_print(" PCA classification score: {}".format(scorePCA))
        compute_KNN_accuracy(X_PCA[:, 0:2], train_labels[:N])

        N = 60000
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=3).fit_transform(np.reshape(train_images, [num_samples, data_size**2*channels])[:N, :])
        scoretsne = compute_KNN_accuracy(X_embedded[:, 0:2], train_labels[:N])


        chief_print(" t-SNE classification score: {}".format(scoretsne))

        plt.figure()
        D = pd.DataFrame({"x": X_embedded[:, 0], "y": X_embedded[:, 1], "color": train_labels})
        sns.scatterplot(data=D, x="x", y="y", hue="color", palette=sns.color_palette("tab10"), legend="brief")
        plt.title(f"tsne, KNN classification accuracy: {scoretsne}")

        plt.savefig(save_dir+"tsne.pdf")
        