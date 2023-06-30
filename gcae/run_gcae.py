"""GenoCAE.

Usage:
  run_gcae.py train --datadir=<name> --data=<name> --model_id=<name> --train_opts_id=<name> --data_opts_id=<name> --save_interval=<num> --epochs=<num> [--resume_from=<num> --trainedmodeldir=<name> --recomb_rate=<num> --superpops=<name> --generations=<num>] [--pheno_model_id=<name>]
  run_gcae.py project --datadir=<name>   [ --data=<name> --model_id=<name>  --train_opts_id=<name> --data_opts_id=<name> --superpops=<name> --epoch=<num> --trainedmodeldir=<name>   --pdata=<name> --trainedmodelname=<name> --alt_data=<name> --recomb_rate=<num> --generations=<num>]
  run_gcae.py plot --datadir=<name> [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py animate --datadir=<name>   [ --data=<name>   --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name> --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py evaluate --datadir=<name> --metrics=<name>  [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name> --alt_data=<name> ]

Options:
  -h --help             show this screen
  --datadir=<name>      directory where sample data is stored. if not absolute: assumed relative to GenoCAE/ directory.
  --data=<name>         file prefix, not including path, of the data files (EIGENSTRAT of PLINK format)
  --trainedmodeldir=<name>     base path where to save model training directories. if not absolute: assumed relative to GenoCAE/ directory. default: ae_out/
  --model_id=<name>     model id, corresponding to a file models/model_id.json
  --train_opts_id=<name>train options id, corresponding to a file train_opts/train_opts_id.json
  --data_opts_id=<name> data options id, corresponding to a file data_opts/data_opts_id.json
  --epochs<num>         number of epochs to train
  --resume_from<num>	saved epoch to resume training from. set to -1 for latest saved epoch.
  --save_interval<num>	epoch intervals at which to save state of model
  --trainedmodelname=<name> name of the model training directory to fetch saved model state from when project/plot/evaluating
  --pdata=<name>     	file prefix, not including path, of data to project/plot/evaluate. if not specified, assumed to be the same the model was trained on.
  --epoch<num>          epoch at which to project/plot/evaluate data. if not specified, all saved epochs will be used
  --superpops<name>     path+filename of file mapping populations to superpopulations. used to color populations of the same superpopulation in similar colors in plotting. if not absolute path: assumed relative to GenoCAE/ directory.
  --metrics=<name>      the metric(s) to evaluate, e.g. hull_error of f1 score. can pass a list with multiple metrics, e.g. "hull_error,f1_score"

  --alt_data=<name> 	project a model on another dataset than it was trained on

"""

from docopt import docopt, DocoptExit
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TF 2.12 prints a lot of redundant warnings.

if "SLURM_NTASKS_PER_NODE" in os.environ:
    if int(os.environ["SLURM_NTASKS_PER_NODE"]) > 1:
        if int(os.environ["SLURM_NTASKS_PER_NODE"]) != len((os.environ["CUDA_VISIBLE_DEVICES"]).split(",")):
            print("Need to have either just 1 process for single-node-multi GPU jobs, or the same number of processes as gpus.")
            exit(3)
        else:
            #If we use more than one task, we need to set devices. For more than 1 process on one node, it will otherwise try to use all gpus on all processes
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import mixed_precision
from datetime import datetime
from utils.data_handler import  get_saved_epochs, get_projected_epochs, write_h5, read_h5, get_coords_by_pop, convex_hull_error, f1_score_kNN, plot_genotype_hist, to_genotypes_sigmoid_round, to_genotypes_invscale_round, GenotypeConcordance, get_pops_with_k, get_baseline_gc, write_metric_per_epoch_to_csv
from utils.visualization import plot_coords_by_superpop, plot_clusters_by_superpop, plot_coords, plot_coords_by_pop, make_animation, write_f1_scores_to_csv,train_test_KDE_plot
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
import utils.layers
import copy
import h5py
import matplotlib.animation as animation
from pathlib import Path
from utils.data_handler import alt_data_generator, parquet_converter
from utils.set_tf_config_berzelius_1_proc_per_gpu import set_tf_config
import shutil
import ContrastiveLosses as CL


if "SLURM_JOBID" in os.environ:
    try:
        shutil.rmtree("./Data/temp"+ os.environ["SLURM_JOBID"])
    except:
        pass
else:
    try:
        shutil.rmtree("./Data/temp")
    except:
        pass

gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpu, True)
k_vec = [1,2,3,4,5]

gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])

enable_mixed_precision = True
if enable_mixed_precision:
    my_type = tf.float16
else:
    my_type = tf.float32


def chief_print(string):

    if "isChief" in os.environ:

        if os.environ["isChief"] == "true":
            print(string)
    else:
        print(string)

def _isChief():

    if "isChief" in os.environ:

        if os.environ["isChief"] == "true":
            return True
        else:
            return False
    else:
        return True


GCAE_DIR = Path(__file__).resolve().parent
class Autoencoder(Model):

    def __init__(self, model_architecture, n_markers, noise_std, regularizer):
        '''

        Initiate the autoencoder with the specified options.
        All variables of the model are defined here.

        :param model_architecture: dict containing a list of layer representations
        :param n_markers: number of markers / SNPs in the data
        :param noise_std: standard deviation of noise to add to encoding layer during training. False if no noise.
        :param regularizer: dict containing regularizer info. False if no regularizer.
        '''
        super(Autoencoder, self).__init__()
        self.all_layers = []
        self.n_markers = n_markers
        self.noise_std = noise_std
        self.residuals = dict()
        self.marker_spec_var = False
        self.passthrough = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.cell_lock = tf.Variable(1.0, dtype=tf.float32, trainable=False)

        self.ge = tf.random.Generator.from_seed(1)

        chief_print("\n______________________________ Building model ______________________________")
        # variable that keeps track of the size of layers in encoder, to be used when constructing decoder.
        ns=[]
        ns.append(n_markers)

        first_layer_def = model_architecture["layers"][0]
        layer_module = getattr(eval(first_layer_def["module"]), first_layer_def["class"])
        layer_args = first_layer_def["args"]
        for arg in ["n", "size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

            if arg in layer_args.keys():
                layer_args[arg] = eval(str(layer_args[arg]))

        if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "flum":
            if "kernel_size" in layer_args:
                dim = 2 * layer_args["kernel_size"] * layer_args["filters"] ** 2
            else:
                dim = layer_args["units"]
            limit = math.sqrt(2 * 3 / (dim))
            layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)


        if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "slimmed":
            dim = layer_args["units"]
            limit = 0.01 # math.sqrt(2 * 3 / (dim))
            layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)



        if "kernel_regularizer" in layer_args and layer_args["kernel_regularizer"] == "L2":

            layer_args["kernel_regularizer"] = tf.keras.regularizers.L2(l2=1e-8)

        try:
            activation = getattr(tf.nn, layer_args["activation"])
            layer_args.pop("activation")
            first_layer = layer_module(activation=activation, **layer_args)

        except KeyError:
            first_layer = layer_module(**layer_args)
            activation = None

        self.all_layers.append(first_layer)
        chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

        if first_layer_def["class"] == "conv1d" and "strides" in layer_args.keys() and layer_args["strides"] > 1:
            ns.append(int(first_layer.shape[1]))
            raise NotImplementedError

        # add all layers except first
        for layer_def in model_architecture["layers"][1:]:
            layer_module = getattr(eval(layer_def["module"]), layer_def["class"])
            layer_args = layer_def["args"]

            for arg in ["n", "size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

                if arg in layer_args.keys():
                    layer_args[arg] = eval(str(layer_args[arg]))

            if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "flum":
                if "kernel_size" in layer_args:
                    dim = layer_args["kernel_size"] * layer_args["filters"]
                else:
                    dim = layer_args["units"]
                limit = math.sqrt(2 * 3 / (dim))
                layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)

            if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "slimmed":
                dim = layer_args["units"]
                limit = 0.01 # math.sqrt(2 * 3 / (dim))
                layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)




            if "kernel_regularizer" in layer_args and layer_args["kernel_regularizer"] == "L2":
                layer_args["kernel_regularizer"] = tf.keras.regularizers.L2(l2=1e-6)

            if "activity_regularizer" in layer_args and layer_args["activity_regularizer"] == "L2":
                layer_args["activity_regularizer"] = tf.keras.regularizers.L2(l2=1e-6 )

            if layer_def["class"] == "MaxPool1D":
                ns.append(int(math.ceil(float(ns[-1]) / layer_args["strides"])))

            if layer_def["class"] == "Conv1D" and "strides" in layer_def.keys() and layer_def["strides"] > 1:
                raise NotImplementedError

            chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

            if "name" in layer_args and (layer_args["name"] == "i_msvar" or layer_args["name"] == "nms"):
                self.marker_spec_var = True

            if "activation" in layer_args.keys():
                activation = getattr(tf.nn, layer_args["activation"])
                layer_args.pop("activation")
                this_layer = layer_module(activation=activation, **layer_args)
            else:
                this_layer = layer_module(**layer_args)

            self.all_layers.append(this_layer)

        if noise_std >0 :
            self.noise_layer = tf.keras.layers.GaussianNoise(noise_std)
        elif noise_std == 0:
            self.noise_layer = tf.keras.layers.GaussianNoise(0.)

        self.ns = ns
        self.regularizer = regularizer

        if self.marker_spec_var:
            random_uniform = tf.random_uniform_initializer()
            self.ms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=my_type))#, name="marker_spec_var")
            self.nms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=my_type))#, name="nmarker_spec_var")

        else:
            chief_print("No marker specific variable.")

    def call(self, input_data, targets=None, is_training = True, verbose = False,  regloss=True, encode_only= False):
        '''
        The forward pass of the model. Given inputs, calculate the output of the model.

        :param input_data: input data
        :param is_training: if called during training
        :param verbose: print the layers and their shapes
        :return: output of the model (last layer) and latent representation (encoding layer)

        '''

        # if we're adding a marker specific variables as an additional channel
        if self.marker_spec_var:
            # Tiling it to handle the batch dimension

            ms_tiled = tf.tile(self.ms_variable, (tf.shape(input_data)[0], 1))
            ms_tiled = tf.expand_dims(ms_tiled, 2)
            nms_tiled = tf.tile(self.nms_variable, (tf.shape(input_data)[0], 1))
            nms_tiled = tf.expand_dims(nms_tiled, 2)
            concatted_input = tf.concat([input_data, ms_tiled], 2)
            input_data = concatted_input

        if verbose:
            chief_print("inputs shape " + str(input_data.shape))

        first_layer = self.all_layers[0]
        counter = 1

        if verbose:
            chief_print("layer {0}".format(counter))
            chief_print("--- type: {0}".format(type(first_layer)))

        x = first_layer(inputs=input_data)

        if "Residual" in first_layer.name:
            out = self.handle_residual_layer(first_layer.name, x, verbose=verbose)
            if not out == None:
                x = out
        if verbose:
            chief_print("--- shape: {0}".format(x.shape))

        # indicator if were doing genetic clustering (ADMIXTURE-style) or not
        have_encoded_raw = False
        encoded_data = None

        # do all layers except first
        for layer_def in self.all_layers[1:]:
            try:
                layer_name = layer_def.cname
            except:
                layer_name = layer_def.name

            counter += 1

            if verbose:
                chief_print("layer {0}: {1} ({2}) ".format(counter, layer_name, type(layer_def)))

            if layer_name == "dropout":
                x = layer_def(x, training = is_training)
            else:
                x = layer_def(x, training = is_training)

            # If this is a clustering model then we add noise to the layer first in this step
            # and the next layer, which is sigmoid, is the actual encoding.
            if layer_name == "encoded_raw":
                encoded_data_pure = x
                encoded_data_pure = tf.keras.layers.Activation('linear', dtype='float32')(x) # This is to ascertain that the output has dtyoe float32. With mixed precision the computations are done in float16.

                #tf.print(tf.reduce_max(tf.math.abs(x)), tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x[:,0]))


                have_encoded_raw = True
                if self.noise_std:
                    x = self.noise_layer(x, training = is_training)

            # If this is the encoding layer, we add noise if we are training
            elif "encoded" in layer_name:

                if encode_only:

                    return x

                if not have_encoded_raw:
                    #tf.print(tf.reduce_max(tf.math.abs(x)), tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x[:,0]))
                    pass
                if self.noise_std and not have_encoded_raw:

                    encoded_data_pure = x
                    encoded_data_pure = tf.keras.layers.Activation('linear', dtype='float32')(x) # This is to ascertain that the output has dtyoe float32. With mixed precision the computations are done in float16.

                    x = self.noise_layer(x, training = is_training)

                encoded_data = x

                #flipsquare = False
                #if self.regularizer and "flipsquare" in self.regularizer:
                #    flipsquare = self.regularizer["flipsquare"]
                
                if "basis" in layer_name:
                    basis = tf.expand_dims(tf.range(-200., 200., 20.), axis=0)
                    x = tf.clip_by_value(tf.concat([tf.expand_dims(x[:,i], axis=1) - basis for i in range(2)], axis=1), -40., 40.)

            if "Residual" in layer_name:
                out = self.handle_residual_layer(layer_name, x, verbose=verbose)
                if not out == None:
                    x = out

            # inject marker-specific variable by concatenation
            if "i_msvar" in layer_name and self.marker_spec_var:
                x = self.injectms(verbose, x, layer_name, ms_tiled, self.ms_variable)

            if "nms" in layer_name and self.marker_spec_var:
                x = self.injectms(verbose, x, layer_name, nms_tiled, self.nms_variable)

            if verbose:
                chief_print("--- shape: {0}".format(x.shape))

        if self.regularizer and regloss and encoded_data is not None:

            if self.regularizer:
                if "box_factor"  in self.regularizer:

                    box_factor = self.regularizer["box_factor"]
                else:
                    box_factor = 0
                if "box_area"  in self.regularizer:
                    box_area = self.regularizer["box_area"]
                else:
                    box_area = 0


            #reg_loss = self.regularizer["reg_factor"] * (box_factor* tf.reduce_sum(tf.math.maximum(tf.constant(0.), tf.square(encoded_data_pure) - 1 * box_area))) #  + 0.* dist_penalty+  0.* tf.reduce_sum(50 * ( -self.cell_lock*0.05 + tf.math.maximum(self.cell_lock*0.05,tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/period))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/period))))))) #  / tf.math.abs(tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/10))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/10)))   )  ))
            #reg_loss = self.regularizer["reg_factor"] * tf.reduce_sum(tf.math.maximum(0., tf.square(encoded_data_pure) - 1 * 40000.))
            reg_loss =  (box_factor* tf.reduce_sum(tf.math.maximum(tf.constant(0.), tf.square(encoded_data_pure) - 1 * box_area))) /box_area / tf.cast(tf.shape(encoded_data_pure)[0], tf.float32) #  + 0.* dist_penalty+  0.* tf.reduce_sum(50 * ( -self.cell_lock*0.05 + tf.math.maximum(self.cell_lock*0.05,tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/period))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/period))))))) #  / tf.math.abs(tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/10))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/10)))   )  ))

            #tf.print(tf.reduce_sum( 1 + 1 * tf.math.maximum(0.0,tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/10))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/10)))))) #  / tf.math.abs(tf.math.sin((encoded_data_pure[:,0] * (2*math.pi/10))) * tf.math.sin((encoded_data_pure[:,1] * (2*math.pi/10)))   )  ))
            #tf.print(tf.reduce_sum(tf.math.maximum(0., tf.square(encoded_data_pure) - 1 * 100)))
            #reg_loss = self.regularizer["reg_factor"] * (tf.reduce_sum(tf.math.maximum(0., tf.square(encoded_data_pure) - 1 * 100)) + 0.1 + 0.1 * tf.math.sin((encoded_data_pure[:,0]) * (2*math.pi/10)) * tf.math.sin((encoded_data_pure[:,1]) * (2*math.pi/10)) / tf.math.abs(5 * tf.math.sin((encoded_data_pure[:,0]) * (2*math.pi/10)) * tf.math.sin((encoded_data_pure[:,1]) * (2*math.pi/10))))

            # if this is a clustering self then the regularization is added to the raw encoding, not the softmaxed one
            #if have_encoded_raw:
            #	reg_loss = reg_func(encoded_data_pure)
            #else:
            #	reg_loss = reg_func(encoded_data)


            #reg_loss += self.regularizer["reg_factor"] * tf.reduce_sum(tf.math.maximum(0., tf.square(x) - 1 * 36.))
            self.add_loss(reg_loss)


        if targets is not None and False:
            reploss = tf.constant(0., tf.float32)
            for i in range(1, tf.shape(encoded_data)[0] - 1):
                shifted = tf.stop_gradient(tf.roll(encoded_data, i, axis=0))
                shifted2 = tf.stop_gradient(tf.roll(encoded_data, i + 1, axis=0))
                shifted_targets = tf.stop_gradient(tf.roll(targets, i, axis=0))
                diff = encoded_data - shifted

                #mean = tf.math.reduce_mean(encoded_data, axis=0, keepdims=True)
                #diff *= tf.expand_dims(tf.where(tf.norm(shifted - mean, axis = -1) < tf.norm(encoded_data - mean, axis = -1), 1.0, 0.0), axis=-1)
                #smalleralong = tf.math.reduce_sum(tf.square(encoded_data - mean), axis = -1) < tf.math.reduce_sum((encoded_data - mean) * (shifted - mean), axis = -1)
                mismatch = tf.math.reduce_mean(tf.where(targets == shifted_targets, 0.0, 1.0), axis=-1)
                #diff *= tf.expand_dims(tf.where(smalleralong, 0.0, 1.0), axis=-1)
                #norm = tf.expand_dims(tf.norm(diff, ord = 2, axis = -1), axis=-1)
                # tf.stop_gradient(diff / (norm + 1e-19)) *
                r2 = (tf.norm(diff, ord = self.regularizer["ord"], axis = -1))**tf.cast(self.regularizer["ord"], tf.float32) + self.regularizer["max_rep"]
                #r2 *= 0.0001
                ##reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * (mismatch * tf.math.exp(-r2 * 0.2)) - 0.02 * tf.math.exp(-r2*0.5*0.2) - 0.02 * tf.math.exp(-r2*0.05*0.2))
                #reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.maximum(0., 0.5 - mismatch * -r2))
                reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.maximum(0., 30. * mismatch - r2))
                shiftedc = (shifted + shifted2)*0.5

                shifteddiff = (shifted - shifted2)

                if False:
                    shifteddiff = tf.stack((-shifteddiff[:,1], shifteddiff[:,0]), axis=1)
                    seconddiff = encoded_data - shiftedc
                    seconddiff = tf.math.mod(diff, 100.)
                    seconddiff += tf.where(diff < -50., 100., 0.)
                    seconddiff += tf.where(diff > 50., -100., 0.)
                    seconddiff *= shifteddiff
                    seconddiff /= tf.norm(shifteddiff) + 1e-9
                else:
                    seconddiff = encoded_data - shiftedc
                    seconddiff -= shifteddiff * tf.math.reduce_sum(seconddiff * shifteddiff, axis=-1, keepdims=True) / (tf.norm(shifteddiff) + 1e-9)**2
                diff = seconddiff
                r2 = (tf.norm(diff, ord = self.regularizer["ord"], axis = -1))**tf.cast(self.regularizer["ord"], tf.float32) + self.regularizer["max_rep"]
                #r2 *= 0.0001
                #reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.exp(-r2 * 0.2))


                #self.add_loss(tf.math.reduce_sum(self.regularizer["rep_factor"] * (tf.math.reduce_mean(tf.where(targets == shifted_targets, 0.0, 1.0), axis=-1) * r2**-6.0 - r2**-3.0)))
                # tf.norm(diff, ord = 2, axis = -1)
                # * f.math.l2_normalize(diff, axis = -1)
            tf.print(reploss)
            self.add_loss(reploss)

        x = tf.keras.layers.Activation('linear', dtype='float32')(x) # This is to ascertain that the output has dtyoe float32. With mixed precision the computations are done in float16.
        encoded_data = tf.keras.layers.Activation('linear', dtype='float32')(encoded_data) # This is to ascertain that the output has dtyoe float32. With mixed precision the computations are done in float16.

        return x, encoded_data


    def handle_residual_layer(self, layer_name, inputs, verbose=False):
        suffix = layer_name.split("Residual_")[-1].split("_")[0]
        res_number = suffix[0:-1]
        if suffix.endswith("a"):
            if verbose:
                chief_print("encoder-to-decoder residual: saving residual {}".format(res_number))
            self.residuals[res_number] = inputs
            return None
        if suffix.endswith("b"):
            if verbose:
                chief_print("encoder-to-decoder residual: adding residual {}".format(res_number))
            residual_tensor = self.residuals[res_number]
            res_length = residual_tensor.shape[1]
            if len(residual_tensor.shape) == 3:
                x = tf.keras.layers.Add()([inputs[:,0:res_length,:], residual_tensor])
            if len(residual_tensor.shape) == 2:
                x = tf.keras.layers.Add()([inputs[:,0:res_length], residual_tensor])

            return x

    def injectms_altered(self, verbose, x, layer_name, ms_tiled, ms_variable):
        """ This version was made to include more tf- functionality. It has not been 100% tested however, but I think it should be okay."""

        if verbose:
            chief_print("----- injecting marker-specific variable")

        # if we need to reshape ms_variable before concatting it
        #if not  tf.shape(x)[1] == self.n_markers :
        d = tf.cast(tf.math.ceil(float(self.n_markers) / tf.cast(tf.shape(x)[1],tf.float32 ) ),tf.int32)
        diff = d*tf.cast(tf.shape(x)[1], tf.int32) - self.n_markers
        ms_var = tf.reshape(tf.pad(ms_variable,[[0,0],[0,diff]]), (-1, tf.shape(x)[1],d))
        # Tiling it to handle the batch dimension
        ms_tiled = tf.tile(ms_var, (tf.shape(x)[0],1,1))

        #else:
        #		# Tiling it to handle the batch dimension
        #		ms_tiled = tf.tile(ms_variable, (tf.shape(x)[0],1))
        #		ms_tiled = tf.expand_dims(ms_tiled, 2)

        if "_sg" in layer_name:
            if verbose:
                chief_print("----- stopping gradient for marker-specific variable")
            ms_tiled = tf.stop_gradient(ms_tiled)


        if verbose:
            chief_print("ms var {}".format(ms_variable.shape))
            chief_print("ms tiled {}".format(ms_tiled.shape))
            chief_print("concatting: {0} {1}".format(x.shape, ms_tiled.shape))

        x = tf.concat([x, ms_tiled], 2)


        return x


    def injectms(self, verbose, x, layer_name, ms_tiled, ms_variable):
        if verbose:
            chief_print("----- injecting marker-specific variable")

        # if we need to reshape ms_variable before concatting it
        if not self.n_markers == x.shape[1]:
            d = int(math.ceil(float(self.n_markers) / int(x.shape[1])))
            diff = d*int(x.shape[1]) - self.n_markers
            ms_var = tf.reshape(tf.pad(ms_variable,[[0,0],[0,diff]]), (-1, x.shape[1],d))
            # Tiling it to handle the batch dimension
            ms_tiled = tf.tile(ms_var, (tf.shape(x)[0],1,1))

        else:
            # Tiling it to handle the batch dimension
            ms_tiled = tf.tile(ms_variable, (x.shape[0],1))
            ms_tiled = tf.expand_dims(ms_tiled, 2)

        if "_sg" in layer_name:
            if verbose:
                chief_print("----- stopping gradient for marker-specific variable")
            ms_tiled = tf.stop_gradient(ms_tiled)


        if verbose:
            chief_print("ms var {}".format(ms_variable.shape))
            chief_print("ms tiled {}".format(ms_tiled.shape))
            chief_print("concatting: {0} {1}".format(x.shape, ms_tiled.shape))

        x = tf.concat([x, ms_tiled], 2)


        return x


def get_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size

    n_samples_last_batch = n_samples % batch_size
    if n_samples_last_batch > 0:
        n_batches += 1
    else:
        n_samples_last_batch = batch_size

    return n_batches, n_samples_last_batch

def alfreqvector(y_pred):
    '''
    Get a probability distribution over genotypes from y_pred.
    Assumes y_pred is raw model output, one scalar value per genotype.

    Scales this to (0,1) and interprets this as a allele frequency, uses formula
    for Hardy-Weinberg equilibrium to get probabilities for genotypes [0,1,2].

    TODO: Currently not current, using logits values in some cases.

    :param y_pred: (n_samples x n_markers) tensor of raw network output for each sample and site
    :return: (n_samples x n_markers x 3 tensor) of genotype probabilities for each sample and site
    '''

    if len(y_pred.shape) == 2:
        alfreq = tf.keras.activations.sigmoid(y_pred)
        alfreq = tf.expand_dims(alfreq, -1)
        return tf.concat(((1-alfreq) ** 2, 2 * alfreq * (1 - alfreq), alfreq ** 2), axis=-1)
        #return tf.concat(((1-alfreq), alfreq), axis=-1)
    else:
        return y_pred[:,:,0:3]#tf.nn.softmax(y_pred)




def generatepheno(data, poplist):
    if data is None:
        return None
    return tf.expand_dims(tf.convert_to_tensor([data.get((fam, name), None) for name, fam in poplist]), axis=-1)

def readpheno(file, num):
    with open(file, "rt") as f:
        for _ in f:
            break
        return {(line[0], line[1]) : float(line[num + 2]) for line in (full_line.split() for full_line in f)}

def writephenos(file, poplist, phenos):
    if _isChief():
        with open(file, "wt") as f:
            for (name, fam), pheno in zip(poplist, phenos):
                f.write(f'{fam} {name} {pheno}\n')


def save_weights2(train_directory, prefix, model):
    if model is None:
        return

    if _isChief():
        if os.path.isdir(prefix):
            newname = train_directory+"_"+str(time.time())
            os.rename(train_directory, newname)
            print("... renamed " + train_directory + " to  " + newname)

        model.save_weights(prefix, save_format ="tf")

def save_weights(prefix, model):
    """
        Apparently, in order to save the model, the save_model call needs to be made on all processes, but they cannot be saved to the same
        file, since that causes a race condition.

    """
    if model is None:
        return
    model.save_weights(prefix, save_format ="tf")


def main():
    chief_print(f"tensorflow version {tf.__version__}")
    tf.keras.backend.set_floatx('float32')

    if enable_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        
    try:
        arguments = docopt(__doc__, version='GenoAE 1.0')
    except DocoptExit:
        chief_print("Invalid command. Run 'python run_gcae.py --help' for more information.")
        exit(1)

    for k in list(arguments.keys()):
        knew = k.split('--')[-1]
        arg=arguments.pop(k)
        arguments[knew]=arg

    if "SLURMD_NODENAME" in os.environ:

        slurm_job = 1
        _, chief, num_workers = set_tf_config()
        isChief = os.environ["SLURMD_NODENAME"] == chief
        os.environ["isChief"] = json.dumps(str(isChief))
        chief_print(num_workers)
        if num_workers > 1  and  not arguments["evaluate"]:
            #Here, NCCL is what I would want to use - it is Nvidias own implementation of reducing over the devices. However, this induces a segmentation fault, and the default value works.
            # had some issues, I think this was resolved by updating to TensorFlow 2.7 from 2.5.
            # However, the image is built on the cuda environment for 2.5 This leads to problems when profiling

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


        else:
            if not isChief:
                print("Work has ended for this worker, now relying only on the Chief :)")
                exit(0)
            #tf.print(tf.config.list_physical_devices(device_type='GPU'))
            #tf.print(tf.test.gpu_device_name())
            num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))

            #chief_print(tf.config.list_physical_devices(device_type='GPU'))
            gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
            #chief_print(gpus)
            strategy =  tf.distribute.MirroredStrategy(devices = gpus, cross_device_ops=tf.distribute.NcclAllReduce())

            slurm_job = 0
        os.environ["isChief"] = json.dumps((isChief))

    else:
        isChief = True
        slurm_job = 0
        num_workers = 1

        strategy =  tf.distribute.MirroredStrategy()
        #chief_print(tf.config.list_physical_devices(device_type='GPU'))

    #chief_print("slurm_job" + str(slurm_job))
    num_devices = strategy.num_replicas_in_sync
    chief_print('Number of devices: {}'.format(num_devices))



    if arguments["trainedmodeldir"]:
        trainedmodeldir = arguments["trainedmodeldir"]
        if not os.path.isabs(trainedmodeldir):
            trainedmodeldir="{}/{}/".format(GCAE_DIR, trainedmodeldir)

    else:
        trainedmodeldir="{}/ae_out/".format(GCAE_DIR)

    if arguments["datadir"]:
        datadir = arguments["datadir"]
        if not os.path.isabs(datadir):
            datadir="{}/{}/".format(GCAE_DIR, datadir)

    else:
        datadir="{}/data/".format(GCAE_DIR)

    if arguments["trainedmodelname"]:
        trainedmodelname = arguments["trainedmodelname"]
        train_directory = trainedmodeldir + trainedmodelname

        split = trainedmodelname.split(".")

        data_opts_id = split[3]
        train_opts_id = split[2]
        model_id = split[1]
        data = split[4]
        pheno_model_id = (split + [None])[5]
    else:
        data = arguments['data']

        data_opts_id = arguments["data_opts_id"]
        train_opts_id = arguments["train_opts_id"]
        model_id = arguments["model_id"]
        pheno_model_id = arguments.get("pheno_model_id")

        train_directory = False


    if arguments["generations"] and arguments["recomb_rate"]:
        generations = int(arguments["generations"])
        recomb_rate = float(arguments["recomb_rate"])

    elif arguments["recomb_rate"] and not arguments["generations"]:
        if  arguments["recomb_rate"] ==0 :
            generations = 0
        else:
            generations = 1
        recomb_rate = float(arguments["recomb_rate"])

    elif not arguments["recomb_rate"] and not arguments["generations"]:

        recomb_rate = 0
        generations = 0
    elif not arguments["recomb_rate"] and  arguments["generations"]:
        chief_print("Cant't recombine {} with no recombination rate")



    with open("{}/data_opts/{}.json".format(GCAE_DIR, data_opts_id)) as data_opts_def_file:
        data_opts = json.load(data_opts_def_file)

    with open("{}/train_opts/{}.json".format(GCAE_DIR, train_opts_id)) as train_opts_def_file:
        train_opts = json.load(train_opts_def_file)

    with open("{}/models/{}.json".format(GCAE_DIR, model_id)) as model_def_file:
        model_architecture = json.load(model_def_file)

    if pheno_model_id is not None:
        with open(f"{GCAE_DIR}/models/{pheno_model_id}.json") as model_def_file:
            pheno_model_architecture = json.load(model_def_file)
    else:
        pheno_model_architecture = None

    for layer_def in model_architecture["layers"]:
        if "args" in layer_def.keys() and "name" in layer_def["args"].keys() and "encoded" in layer_def["args"]["name"] and "units" in layer_def["args"].keys():
            n_latent_dim = layer_def["args"]["units"]
    tf.print("LATENT DIM: ", n_latent_dim)
    # indicator of whether this is a genetic clustering or dimensionality reduction model
    doing_clustering = False
    for layer_def in model_architecture["layers"][1:-1]:
        if "encoding_raw" in layer_def.keys():
            doing_clustering = True

    tf.print("clustering: ", doing_clustering)

    chief_print("\n______________________________ arguments ______________________________")
    for k in arguments.keys():
        chief_print(k + " : " + str(arguments[k]))
    chief_print("\n______________________________ data opts ______________________________")
    for k in data_opts.keys():
        chief_print(k + " : " + str(data_opts[k]))
    chief_print("\n______________________________ train opts ______________________________")
    for k in train_opts:
        chief_print(k + " : " + str(train_opts[k]))
    chief_print("______________________________")


    batch_size = train_opts["batch_size"] * num_devices
    #local_batch_size  = train_opts["batch_size"]
    learning_rate = train_opts["learning_rate"]  * num_devices
    regularizer = train_opts["regularizer"]

    superpopulations_file = arguments['superpops']
    if superpopulations_file and not os.path.isabs(os.path.dirname(superpopulations_file)):
        superpopulations_file="{}/{}/{}".format(GCAE_DIR, os.path.dirname(superpopulations_file), Path(superpopulations_file).name)

    norm_opts = data_opts["norm_opts"]
    norm_mode = data_opts["norm_mode"]
    validation_split = data_opts["validation_split"]


    try:
        holdout_val_pop = data_opts["holdout_val_pop"]
    except:

        holdout_val_pop = None


    if "sparsifies" in data_opts.keys():
        missing_mask_input = True
        sparsifies = data_opts["sparsifies"]

    else:
        missing_mask_input = False

    if "impute_missing" in data_opts.keys():
        fill_missing = data_opts["impute_missing"]

    else:
        fill_missing = False

    if fill_missing:
        chief_print("Imputing originally missing genotypes to most common value.")
    else:
        chief_print("Keeping originally missing genotypes.")
        missing_mask_input = True

    if not train_directory:
        dirparts = [model_id, train_opts_id, data_opts_id, data] + ([pheno_model_id] if pheno_model_id is not None else [])
        train_directory = trainedmodeldir + "ae." + ".".join(dirparts)

    if arguments["alt_data"]:
        alt_data = arguments["alt_data"]
        data = arguments["alt_data"]
    else:
        alt_data = None

    if arguments["pdata"]:
        pdata = arguments["pdata"]
    else:
        pdata = data


    data_prefix = datadir + pdata


    results_directory = "{0}/{1}".format(train_directory, pdata)
    if alt_data is not None:
        results_directory = train_directory+"/"+str(alt_data)
    try:
        os.makedirs(train_directory,exist_ok = True)
    except OSError:
        pass
    try:
        os.mkdir(results_directory)
    except OSError:
        pass

    encoded_data_file = "{0}/{1}/{2}".format(train_directory, pdata, "encoded_data.h5")
    #encoded_data_file = "{0}/{1}".format(results_directory, "encoded_data.h5")

    if "noise_std" in train_opts.keys():
        noise_std = train_opts["noise_std"]
    else:
        noise_std = False

    if "decode_only" in data_opts.keys():
        decode_only = data_opts["decode_only"]
    else:
        decode_only = False


    if (arguments['evaluate'] or arguments['animate'] or arguments['plot']):

        if os.path.isfile(encoded_data_file):
            #encoded_data = h5py.File(encoded_data_file, 'r')
            pass
        else:
            print("------------------------------------------------------------------------")
            print("Error: File {0} not found.".format(encoded_data_file))
            print("------------------------------------------------------------------------")
            exit(1)

        epochs = get_projected_epochs(encoded_data_file)

        if arguments['epoch']:
            epoch = int(arguments['epoch'])
            if epoch in epochs:
                epochs = [epoch]
            else:
                print("------------------------------------------------------------------------")
                print("Error: Epoch {0} not found in {1}.".format(epoch, encoded_data_file))
                print("------------------------------------------------------------------------")
                exit(1)

        if doing_clustering:
            if arguments['animate']:
                print("------------------------------------------------------------------------")
                print("Error: Animate not supported for genetic clustering model.")
                print("------------------------------------------------------------------------")
                exit(1)


            if arguments['plot'] and not superpopulations_file:
                print("------------------------------------------------------------------------")
                print("Error: Plotting of genetic clustering results requires a superpopulations file.")
                print("------------------------------------------------------------------------")
                exit(1)

    else:
        max_mem_size = 5 * 10**10 # this is approx 50GB
        if alt_data is not None:
            data_prefix = datadir + alt_data

        filebase = data_prefix
        parquet_converter(filebase, max_mem_size=max_mem_size)

        if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
            n_train_samples_for_data = int(train_opts["n_samples"])
        else:
            n_train_samples_for_data = -1

        data = alt_data_generator(filebase= data_prefix,
                        batch_size = batch_size,
                        normalization_mode = norm_mode,
                        normalization_options = norm_opts,
                        impute_missing = fill_missing,
                        generations = generations,
                        recombination_rate= 0.0, #recomb_rate,
                        only_recomb= False,
                        n_samples = n_train_samples_for_data)
        chief_print("Recombination rate: " + str(data.recombination_rate))

        n_markers = data.n_markers
        if holdout_val_pop is None:
            data.define_validation_set2(validation_split= validation_split)
        else:
            data.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)


        data.missing_mask_input = missing_mask_input

        n_unique_train_samples = copy.deepcopy(data.n_train_samples)
        n_train_samples = n_unique_train_samples

        n_valid_samples = copy.deepcopy(data.n_valid_samples)

        batch_size_valid = batch_size
        n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size)
        _, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)
        data.n_train_samples_last_batch = int(n_train_samples_last_batch)
        data.n_valid_samples_last_batch = int(n_valid_samples_last_batch)
        data.n_train_samples = n_train_samples
        data.sample_idx_train = np.arange(n_train_samples)

        if holdout_val_pop is None:
            data.define_validation_set2(validation_split= validation_split)
        else:
            data.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)

        n_markers = data.n_markers
        if pheno_model_architecture is not None:
            phenodata = readpheno(data_prefix + ".phe", 2)
        else:
            phenodata = None

        chunk_size = 5*data.batch_size
        ds = data.create_dataset_tf_record(chunk_size, "training", num_workers)
        if n_valid_samples > 0:

            ds_validation = data.create_dataset_tf_record(chunk_size, "validation", num_workers)

        a0 = 0
        a1 = 0
        a2 = 0
        a0_valid = 0
        a1_valid = 0
        a2_valid = 0

        for batch_dist_input, batch_dist_target, poplist,_,_ in ds:
            a0 += tf.reduce_sum(tf.cast((2*batch_dist_target==0),tf.float32),axis=0).numpy()
            a1 += tf.reduce_sum(tf.cast((2*batch_dist_target==1),tf.float32),axis=0).numpy()
            a2 += tf.reduce_sum(tf.cast((2*batch_dist_target==2),tf.float32),axis=0).numpy()

        if n_valid_samples > 0:
            for batch_dist_input, batch_dist_target, poplist, _, _ in ds_validation:
                a0_valid += tf.reduce_sum(tf.cast((2*batch_dist_target==0),tf.float32),axis=0).numpy()
                a1_valid += tf.reduce_sum(tf.cast((2*batch_dist_target==1),tf.float32),axis=0).numpy()
                a2_valid += tf.reduce_sum(tf.cast((2*batch_dist_target==2),tf.float32),axis=0).numpy()

        A = tf.transpose(tf.stack([a0,a1,a2]))

        A_valid = tf.transpose(tf.stack([a0_valid,a1_valid,a2_valid]))


        maf = (1*a1+ 2*a2 + 1*a1_valid + 2*a2_valid) / 2.0 / (a0 + a1 + a2 + a0_valid + a1_valid + a2_valid)

        bins = 1/100 * np.array([2,5,10,15,20,25,30,40,50,60])[:,np.newaxis]

        bins = 1/100 * np.array([2,4,8,16,32,50,60])[:,np.newaxis]


        maf2 = maf[tf.newaxis,:]
        bin_inds = len(bins) - np.sum(maf2<bins,0)


        loss_def = train_opts["loss"]

        if loss_def["module"] == "tf.keras.losses":

            loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
            if "args" in loss_def.keys():
                loss_args = loss_def["args"]
            else:
                loss_args = dict()
            loss_obj = loss_class(**loss_args)
            contrastive = False
        else:
            loss_class = loss_def["module"]+"_"+(loss_def["class"])
            contrastive = True


        def get_originally_nonmissing_mask(genos):
            '''
            Get a boolean mask representing missing values in the data.
            Missing value is represented by float(norm_opts["missing_val"]).

            Uses the presence of missing_val in the true genotypes as indicator, missing_val should not be set to
            something that can exist in the data set after normalization!!!!

            :param genos: (n_samples x n_markers) genotypes
            :return: boolean mask of the same shape as genos
            '''
            orig_nonmissing_mask = tf.not_equal(genos, float(norm_opts["missing_val"]))

            return orig_nonmissing_mask

        with strategy.scope():


            if not contrastive:
                if loss_class == tf.keras.losses.CategoricalCrossentropy or loss_class == tf.keras.losses.KLDivergence :


                    beta = 0.999
                    #Beta = (1.0 - beta) / (1-tf.math.pow(beta,tf.cast(A[:n_markers,:],tf.float32)+1))
                    #alpha = 3 / (tf.reduce_sum(Beta,axis = -1 ,keepdims = True))
                    #coeff = alpha*Beta

                    def loss_func(y_pred, y_true, pow=1., avg=False):
                        y_pred = y_pred[:, 0:n_markers]

                        #if not fill_missing:
                        #	orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)
                        #else:
                        #	orig_nonmissing_mask = np.full(y_true.shape, True)
                        # TODO: Reintroduce missingness support here, with proper shape after slicing!

                        y_pred = alfreqvector(y_pred)
                        y_true = tf.one_hot(tf.cast(y_true * 2, tf.uint8), 3) * 0.9997 + 0.0001

                        y_true2 = y_true#*0.997 + 0.001
                        if avg:
                            y_true = y_true * -1./tf.cast(tf.shape(y_true)[0], tf.float32) + tf.math.reduce_mean(y_true, axis=0, keepdims=True)

                        y_pred *= pow
                        y_pred = y_pred - tf.stop_gradient(tf.math.reduce_max(y_pred, axis=-1, keepdims=True))
                        y_pred_prob = tf.nn.softmax(y_pred)

                        gamma = 1.0

                        #partialres = - (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true * (1 - tf.math.reduce_sum(tf.stop_gradient(y_pred_prob) * y_true, axis=-1, keepdims=True)/tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True))**gamma, axis = 0) * (1.0 - beta) / (1-tf.math.pow(beta,tf.cast(A[:n_markers,:],tf.float32)+1)+1e-9))

                        # This line is my implementation of the class balanced based loss
                        #partialres = - (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true * (1 - tf.math.reduce_sum(tf.stop_gradient(y_pred_prob) * y_true, axis=-1, keepdims=True)/tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True))**gamma, axis = 0) * coeff)

                        # Vanilla CCE, multiply by coeff for Class balance based loss. Summing over samples, mean over markers
                        #partialres =  -tf.reduce_sum(y_true * tf.math.log(y_pred_prob+ 10**-100),axis = 0) # *coeff

                        #This line below is the "original"
                        partialres = - (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true * (1 - tf.math.reduce_sum(tf.stop_gradient(y_pred_prob) * y_true, axis=-1, keepdims=True)/tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True))**gamma, axis = 0) * tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)-1))

                        #This line below is a test, don't use
                        #partialres = - tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True)))) /tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True)
                        return tf.math.reduce_mean(partialres)

                else:
                    def loss_func(y_pred, y_true):

                        y_pred = y_pred[:, 0:n_markers]
                        y_true = tf.convert_to_tensor(y_true)

                        if not fill_missing:
                            orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)
                            y_pred = y_pred[orig_nonmissing_mask]
                            y_true = y_true[orig_nonmissing_mask]

                        return loss_obj(y_pred = y_pred, y_true = y_true)
            else:

                # Instantiate the contrastive loss functions, implementations in the ContrastiveLosses module
                loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
                if "args" in loss_def.keys():
                    loss_args = loss_def["args"]
                else:
                    loss_args = dict()
                loss_func = loss_class(**loss_args)


            @tf.function
            def distributed_train_step(model, model2, optimizer, optimizer2, loss_function, input_data, targets, pure, phenomodel=None, phenotargets=None,data=None):

                per_replica_losses, local_num_total_k_mer, local_num_correct_k_mer, local_class_conc  = strategy.run(run_optimization, args=(model, model2, optimizer, optimizer2, loss_function, input_data, targets, pure, phenomodel, phenotargets,data))
                loss = strategy.reduce("SUM", per_replica_losses, axis=None)
                num_total_k_mer = strategy.reduce("SUM", local_num_total_k_mer, axis=None)
                num_correct_k_mer = strategy.reduce("SUM", local_num_correct_k_mer, axis=None)

                class_conc = strategy.reduce("SUM", local_class_conc, axis=None)

                return loss,  num_total_k_mer,num_correct_k_mer,class_conc


            @tf.function
            def valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data):

                output_valid_batch, _ = autoencoder(input_valid_batch, is_training = True, regloss=False)
                if contrastive:

                    valid_loss_batch = loss_func(anchors = output_valid_batch[:tf.shape(output_valid_batch)[0] // 2, :], positives = output_valid_batch[tf.shape(output_valid_batch)[0] // 2:, :])

                    #valid_loss_batch = loss_func(y_pred=output_valid_batch, y_true=targets_valid_batch)
                    num_total_k_mer, num_correct_k_mer = tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
                    class_conc = tf.reshape(tf.convert_to_tensor([0., 0., 0.]), [1, 3])
                else:
                    valid_loss_batch = loss_func(y_pred=output_valid_batch, y_true=targets_valid_batch)
                    num_total_k_mer, num_correct_k_mer = compute_k_mer_concordance(targets_valid_batch, output_valid_batch, data)
                    class_conc = compute_class_concordance(targets_valid_batch, output_valid_batch)

                return valid_loss_batch, num_total_k_mer, num_correct_k_mer,class_conc

            @tf.function
            def proj_batch(model, loss_func, input_valid_batch, targets_valid_batch,data,poplist,poplist_batch):
                output, encoded_data = model(input_valid_batch, is_training=False, regloss=False)

                if contrastive:
                    valid_loss_batch = 0# Does not really make sense to compute this, since it depends on the augmentations.
                    num_total_k_mer, num_correct_k_mer = tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
                    class_conc = tf.reshape(tf.convert_to_tensor([0., 0., 0.]), [1, 3])
                else:
                    valid_loss_batch = loss_func(y_pred = output, y_true = targets_valid_batch)
                    num_total_k_mer, num_correct_k_mer =  compute_k_mer_concordance(targets_valid_batch, output, data)
                    class_conc = compute_class_concordance(targets_valid_batch, output)

                try:
                    # This wont work if first dimension is empty
                    pop_inds = tf.cast(tf.where(poplist[:,1] == poplist_batch[:,0,tf.newaxis]) [:,1], tf.float32)

                except:
                    pop_inds = tf.cast(tf.reshape(tf.convert_to_tensor([]), [0]), tf.float32)


                return valid_loss_batch, num_total_k_mer, num_correct_k_mer,class_conc,output,encoded_data,pop_inds, targets_valid_batch

            @tf.function
            def distributed_valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data):

                per_replica_losses, local_num_total_k_mer, local_num_correct_k_mer,local_class_conc   = strategy.run(valid_batch, args=(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data))

                loss = strategy.reduce("SUM", per_replica_losses, axis=None)
                num_total_k_mer = strategy.reduce("SUM", local_num_total_k_mer, axis=None)
                num_correct_k_mer = strategy.reduce("SUM", local_num_correct_k_mer, axis=None)
                class_conc = strategy.reduce("SUM", local_class_conc, axis=None)

                return loss, num_total_k_mer, num_correct_k_mer, class_conc

            @tf.function
            def distributed_proj_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data,poplist,poplist_batch):

                per_replica_losses, local_num_total_k_mer, local_num_correct_k_mer,local_class_conc, decoded_data_valid_batch, encoded_data_valid_batch,pop_inds_batch, targets_valid_batch = strategy.run(proj_batch, args=(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data,poplist,poplist_batch))
                loss = strategy.reduce("SUM", per_replica_losses, axis=None)
                num_total_k_mer = strategy.reduce("SUM", local_num_total_k_mer, axis=None)
                num_correct_k_mer = strategy.reduce("SUM", local_num_correct_k_mer, axis=None)
                class_conc = strategy.reduce("SUM", local_class_conc, axis=None)
                encoded_data =  strategy.gather(tf.cast(encoded_data_valid_batch,tf.float32), axis=0)
                decoded_data = strategy.gather(decoded_data_valid_batch, axis=0)
                pop_inds = strategy.gather(pop_inds_batch, axis=0)
                targets = strategy.gather(targets_valid_batch, axis=0)


                return loss, num_total_k_mer, num_correct_k_mer, class_conc,decoded_data, encoded_data,pop_inds,targets

            @tf.function
            def compute_concordance(inputs, truth, prediction, data):
                """
                I want to compute the genotype concordance of the prediction when using multi-gpu training.

                Compute them for each batch, and at the end of the epoch just average over the batches?
                Want to only compute the concordance for non-missing data

                """

                if train_opts["loss"]["class"] == "MeanSquaredError" and (data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard") :

                    genotypes_output = to_genotypes_invscale_round(prediction[:, 0:n_markers], scaler_vals = [data.scaler.mean_, data.scaler.var_])
                    true_genotypes = to_genotypes_invscale_round(truth, scaler_vals = [data.scaler.mean_, data.scaler.var_])

                elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01" or train_opts["loss"]["module"] == "contrastive":
                    genotypes_output = to_genotypes_sigmoid_round(prediction[:, 0:n_markers])
                    true_genotypes = truth

                elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts["norm_mode"] == "genotypewise01":
                    genotypes_output = tf.cast(tf.argmax(alfreqvector(prediction[:, 0:n_markers]), axis = -1), tf.float32) * 0.5
                    true_genotypes = truth

                else:
                    chief_print("Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(train_opts["loss"]["class"],data_opts["norm_mode"]))
                    true_genotypes = truth
                    #genotypes_output = np.array([])
                    #true_genotypes = np.array([])


                use_indices = tf.where(inputs[:,:, 1] !=  2  )
                tf.print(inputs)
                diff = true_genotypes - genotypes_output
                diff2 = tf.gather_nd(diff, indices=use_indices)
                num_total = tf.cast(tf.shape(diff2)[0], tf.float32)
                num_correct = tf.cast(tf.shape(tf.where(diff2 == 0))[0], tf.float32)

                return num_total, num_correct

            @tf.function
            def compute_class_concordance(truth,prediction):
                """
                    Here we compute the class concordance. This is done by first changing from the output to predicted
                    genotype by the function alfreqvector. We then one hot transform both target and output (truth and
                    prediction). Here truth has to be multipled by two, this is done due to the normalization.

                    NOTE: This normalization is done only when the 0,0.5,1 normalization (genotypewise01) is done, otherwise, we need to
                    do the inverse mapping of the used normalization.


                    Then we use a one-hot encoding, adding the two should yield a 2 in a positon where the truth and the
                    prediction matches.

                    Then, sum over axis 0, which is the batch axis, and then sum over -1, which should be the marker
                    dimension.

                    This then yields a three-dimensional vector, containing number of correct predictions for each class.

                    This quantity then has to be summed over all batches, and scaled by the number of samples used, as
                    well as the marker count.

                    Now, I would also be interested in the number of wrong predictions, in order to get a grasp on
                    how accurately it predcits, not only shooting

                """

                y_pred = tf.one_hot(tf.cast(tf.argmax(alfreqvector(prediction[:, 0:n_markers]), axis=-1),tf.uint8), depth = 3, axis = -1)

                y_true = tf.one_hot(tf.cast(truth * 2, tf.uint8), 3)

                return tf.reduce_sum(tf.reduce_sum(tf.cast((y_pred+y_true) == 2,tf.float32),axis = 1),axis = 0)

            @tf.function
            def compute_k_mer_concordance(truth,prediction, data):
                """
                I want to compute the genotype concordance of the prediction when using multi-gpu training.

                Compute them for each batch, and at the end of the epoch just average over the batches?
                Want to only compute the concordance for non-missing data

                Seems like the shape here is batch_size *n_markers.
                """

                if train_opts["loss"]["class"] == "MeanSquaredError" and (
                        data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):

                    genotypes_output = to_genotypes_invscale_round(prediction[:, 0:n_markers],
                                                                scaler_vals=[data.scaler.mean_, data.scaler.var_])
                    true_genotypes = to_genotypes_invscale_round(truth,
                                                                scaler_vals=[data.scaler.mean_, data.scaler.var_])

                elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
                    genotypes_output = to_genotypes_sigmoid_round(prediction[:, 0:n_markers])
                    true_genotypes = truth

                elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts[
                    "norm_mode"] == "genotypewise01":
                    genotypes_output = tf.cast(tf.argmax(alfreqvector(prediction[:, 0:n_markers]), axis=-1),
                                            tf.float32) * 0.5
                    true_genotypes = truth

                else:
                    chief_print(
                        "Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(
                            train_opts["loss"]["class"],
                            data_opts["norm_mode"]))


                num_total = []
                num_correct = []


                for k in k_vec:


                    # I am stupid to not document these lines better when writing them. What do they do?
                    # Reconstructing what I think I did, so I am adding this last element as a dummy-marker, only to make the tensor slicing easier in the next step
                    # I want to be able to extract elements like vec[x:y], where I might want to access the last element. To do that I would need to do vec[x:], since vec[x:-1]
                    # Skips the last element.

                    # This try statement had to be done since the distribution strategy may send a batch with 0 samples, for some cases when using the tfrecord, and manual sharding
                    # Not a very good fix, but works for this purpose.
                    try:
                        truth2 = tf.transpose(tf.concat([tf.cast(true_genotypes, tf.float32), -1.0 * tf.ones(shape=[tf.shape(truth)[0], 1])], axis=1))


                        prediction2 = tf.transpose(
                            tf.concat([tf.cast(genotypes_output, tf.float32), -1.0 * tf.ones(shape=[tf.shape(truth)[0], 1])],
                                    axis=1))

                        # Here the dimensions are [n_markers+1,n_samples]

                        # The following lines  results in a tensor of size (n_markers-k+1, k, n_samples), where we sort of gather the different k-mers for all samples.

                        d1 = tf.stack([truth2[i:-k + i, :] for i in range(k)], axis=1)
                        d2 = tf.stack([prediction2[i:-k + i, :] for i in range(k)], axis=1)

                        # Check for how many of the k-mers are fully matched, i.e the reduced sum of the difference should be zero in axis 1.
                        diff = tf.math.reduce_sum(
                            tf.math.reduce_sum(tf.cast(tf.math.reduce_sum(tf.math.abs(d1 - d2), axis=1) == 0, tf.float32),
                                            axis=0))


                        num_total_temp = tf.cast(tf.shape(d1)[0] * tf.shape(d1)[2],tf.float32) # n_markers * n_samples in this batch.
                        num_correct_temp = tf.cast(diff,tf.float32) # unsure why this needs to be cast
                        num_total.append(num_total_temp)
                        num_correct.append(num_correct_temp)

                    except:
                        num_total.append(tf.cast(0,tf.float32))
                        num_correct.append(tf.cast(0,tf.float32))


                # Convert from lists to tensors.
                num_correct = tf.stack(num_correct, axis=0)
                num_total = tf.stack(num_total, axis=0)


                num_correct = tf.reshape(num_correct, [len(k_vec),1])
                num_total = tf.reshape(num_total,  [len(k_vec),1])

                return num_total, num_correct

            @tf.function
            def distributed_compute_concordance(input, truth, prediction,data):

                local_num_total, local_num_correct = strategy.run(compute_concordance, args = (input, truth, prediction,data))

                num_total = strategy.reduce("SUM", local_num_total, axis=None)
                num_correct = strategy.reduce("SUM", local_num_correct, axis=None)

                return num_total, num_correct


            @tf.function
            def run_optimization_pareto(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel=None, phenotargets=None,data = None):
                '''

                Currently not in use, but wights the gradients from different loss contributions differentyl.

                Run one step of optimization process based on the given data.

                :param model: a tf.keras.Model
                :param optimizer: a tf.keras.optimizers
                :param loss_function: a loss function
                :param input: input data
                :param targets: target data
                :return: value of the loss function
                '''

                full_loss = True
                do_two = False
                do_softmaxed = False
                with tf.GradientTape() as g:
                    output, encoded_data = model(input, targets, is_training=True, regloss=False)

                    if pure and phenomodel is not None:
                        z = phenomodel(encoded_data, is_training=True)
                        loss_value = tf.reduce_sum(z[0])
                    if pure or full_loss:


                        if contrastive:
                            loss_value = loss_function(anchors = output[:tf.shape(output)[0] // 2, :], positives = output[tf.shape(output)[0] // 2:, :])
                        else:
                            loss_value = loss_function(y_pred = output, y_true = targets)

                        if do_two:
                            output2, _ = model2(input, targets, is_training=True, regloss=False)
                            loss_value += loss_function(y_pred = output2, y_true = targets)

                if contrastive:
                    num_total_k_mer, num_correct_k_mer  =tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
                    class_conc = tf.reshape(tf.convert_to_tensor([0., 0., 0.]), [1, 3])
                else:

                    num_total_k_mer, num_correct_k_mer = compute_k_mer_concordance(targets, output, data)

                    class_conc = compute_class_concordance(targets, output)


                allvars = model.trainable_variables + (model2.trainable_variables if model2 is not None else []) + (phenomodel.trainable_variables if phenomodel is not None else [])
                gradients = g.gradient(loss_value, allvars)
                orig_loss = loss_value

                with tf.GradientTape() as g5:
                    loss_value = tf.constant(0.)
                    if do_softmaxed:
                        for output, encoded_data in (model(input, targets, is_training=True, regloss=False),) + ((model2(input, targets, is_training=True, regloss=False), ) if do_two else ()):
                            y_true = tf.one_hot(tf.cast(targets * 2, tf.uint8), 3)
                            y_pred = tf.nn.softmax(output[:,0:model.n_markers])
                            loss_value += tf.math.reduce_sum(((-y_pred) * y_true)) * 1e-6
                    for val in allvars:
                        #maxflax = tf.math.reduce_max(tf.math.abs(val))
                        maxflax2 = tf.math.minimum(tf.math.reduce_max(tf.math.abs(val)), tf.math.reduce_max(tf.math.abs(1. - tf.math.abs(val))))
                        loss_value += tf.square(tf.math.maximum(1.0, maxflax2))
                    #if pure or full_loss:
                    #	loss_value = -loss_function(y_pred = output, y_true = targets, avg=True)

                    #else:

                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))
                    #maxflaxloss = loss_value

                #### gradients_maxflaxloss = g5.gradient(maxflaxloss, allvars)
                gradientsavg = g5.gradient(loss_value, allvars)
                #other_loss4 = loss_value
                #other_loss4 = 0

                #with tf.GradientTape() as g4:
                #	output, encoded_data = model(input, targets, is_training=True, rander=[False, True])
                #	if pure or full_loss:
                #		loss_value = loss_function(y_pred = output, y_true = targets)

                    #else:

                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

                #gradientsrandx = g4.gradient(loss_value, model.trainable_variables)
                #other_loss3 = loss_value
                #with tf.GradientTape() as g4:
                #	output, encoded_data = model(input, targets, is_training=True)
                #	if pure or full_loss:
                #		loss_value = loss_function(y_pred = output, y_true = targets, pow=2.)

                    #else:

                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
                    #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

                #gradientssq = g4.gradient(loss_value, model.trainable_variables)
                #other_loss3 = loss_value

                with tf.GradientTape() as g4:
                    output, encoded_data = model(input, targets, is_training=True)
                    loss_value = sum(model.losses)
                    if do_two:
                        output2, encoded_data2 = model2(input, targets, is_training=True)
                        loss_value += sum(model2.losses)
                gradientsc = g4.gradient(loss_value, allvars)
                #other_loss2 = loss_value

                if do_two:
                    factor = 0.
                    with tf.GradientTape() as g2:
                        output, encoded_data = model(input, targets, is_training=True, regloss=False)
                        output2, encoded_data2 = model2(input, targets, is_training=True, regloss=False)
                        loss_value = tf.math.reduce_sum( -tf.math.log(0.5+0.5*tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0))
                        * (factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0)), axis=-1)
                        * tf.math.rsqrt
                        (tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0)) * (factor*encoded_data-tf.roll(encoded_data, 1, axis=0)), axis=-1) * tf.reduce_sum(
                        (factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0)) * (factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0))+1e-4, axis=-1))))*1e-2
                    gradients2 = g2.gradient(loss_value, allvars)
                    #other_loss = loss_value

                    with tf.GradientTape() as g3:
                        output, encoded_data = model(input, targets, is_training=True, regloss=False)
                        output2, encoded_data2 = model2(input, targets, is_training=True, regloss=False)
                        loss_value = tf.math.reduce_sum( -tf.math.log(1.-0.5*tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0))
                        * (factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0)), axis=-1)
                        * tf.math.rsqrt
                        (tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0)) * (factor*encoded_data-tf.roll(encoded_data, 1, axis=0)), axis=-1) * tf.reduce_sum(
                        (factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0)) * (factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0))+1e-4, axis=-1))))*1e-2
                    ##	output, encoded_data = model(input, targets, is_training=True)
                    ##	#loss_value = loss_function(y_pred = output, y_true = targets) * (1.0 if pure or full_loss else 0.0)
                        #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
                        #loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))
                    ##	y_pred = output - tf.stop_gradient(tf.math.reduce_max(output, axis=-1, keepdims=True))
                    ##	loss_value = -tf.math.reduce_mean(tf.square(y_pred - tf.roll(y_pred, 1, axis = 0)) * 1e-3)
                    gradientsb = g3.gradient(loss_value, allvars)
                    #other_loss3 = loss_value

                if phenomodel is not None:
                    with tf.GradientTape() as g6:
                        loss_value = tf.constant(0.)
                        for output, encoded_data in (model(input, targets, is_training=True, regloss=False),) + ((model2(input, targets, is_training=True, regloss=False), ) if do_two else ()):
                            phenopred, _ = phenomodel(encoded_data, is_training=True)
                            tf.print("PRED")
                            tf.print(phenopred)
                            tf.print(phenotargets)
                            loss_value += tf.math.reduce_sum(tf.square(phenopred - phenotargets)) * 1e-2

                    gradientspheno = g6.gradient(loss_value, allvars)
                    #phenoloss = loss_value


                loss_value = orig_loss

                ##radients3 = []

                ##loss_value += orig_loss

                ##for g1, g2 in zip(gradients, gradients2):
                ##	if g1 is None:
                ##		g3 = g2
                ##	elif g2 is None:
                ##		g3 = g1
                ##	else:
                ##		#g3 = tf.where(tf.math.sign(g1 * g2) >= 0, g1 + g2, 0.)
                ##		summed = g1 + g2
                ##		g3 = tf.where(tf.math.sign(g1 * g2) >= 0, summed, tf.math.sign(summed) * tf.math.minimum(0.5 * tf.math.minimum(tf.abs(g1), tf.abs(g2)), tf.abs(summed)))
                ##	gradients3.append(g3)
                def combine(gradients, gradients2):
                    alphanom = tf.constant(0.)
                    alphadenom = tf.constant(1.0e-30)
                    for g1, g2 in zip(gradients, gradients2):
                        if g1 is not None and g2 is not None:
                            gdiff = g2 - g1
                            alphanom += tf.math.reduce_sum(gdiff * g2)
                            alphadenom += tf.math.reduce_sum(gdiff * gdiff)
                    alpha = alphanom / alphadenom
                    gradients3 = []
                    cappedalpha = tf.clip_by_value(alpha, 0., 1.)
                    for g1, g2 in zip(gradients, gradients2):
                        if g1 is None:
                            gradients3.append(g2)
                        elif g2 is None:
                            gradients3.append(g1)
                        else:
                            gradients3.append(g1 * (1-cappedalpha) + g2 * (cappedalpha))
                    return (gradients3, alpha)

                #gradients4, alpha4 = combine(gradientsrandx, gradientsrandy)
                #alpha4 = 0
                #gradients3, alpha4 = combine(gradients_output_loss, gradients_maxflaxloss)
                gradients3, alpha4 = combine(gradients, gradientsavg)

                alpha3 = 0
                #gradients4 = gradientsrandx
                if do_two:
                    #gradients4, alpha3 = combine(gradients2, gradientsb)
                    #gradients3, alpha = combine(gradients3, gradients4)
                    pass
                else:
                    alpha3 = 0
                    alpha = 0
                    other_loss3 = 0
                    other_loss = 0

                #gradients3, alpha2 = combine(gradients3, gradient_model_losses)
                gradients3, alpha2 = combine(gradients3, gradientsc)
                tf.print("alpha for reg-losses: ", alpha2)
                if phenomodel is not None:
                    gradients3, phenoalpha = combine(gradients3, gradientspheno)
                else:
                    phenoloss, phenoalpha = (0.,0.)
                if pure or full_loss:
                    optimizer.apply_gradients(zip(gradients3, allvars))
                #if pure or not full_loss:
                #	# was optimizer2
                #	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                #tf.print(loss_value, other_loss4, other_loss3, other_loss2, other_loss, full_loss, alpha4, alpha3, alpha2, alpha)
                #tf.print(loss_value, other_loss4, other_loss3, other_loss2, other_loss, phenoloss, full_loss, alpha4, alpha3, alpha2, alpha, phenoalpha)



                return loss_value, num_total_k_mer, num_correct_k_mer, class_conc
            @tf.function
            def run_optimization(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel=None, phenotargets=None,data = None):
                '''
                Run one step of optimization process based on the given data, basic version.
                    

                :param model: a tf.keras.Model
                :param optimizer: a tf.keras.optimizers
                :param loss_function: a loss function
                :param input: input data
                :param targets: target data
                :return: value of the loss function
                '''

                with tf.GradientTape() as g:
                    output, _ = model(input, is_training=True)
                    if contrastive:
                        loss_value = loss_function(anchors = output[:tf.shape(output)[0] // 2, :], positives = output[tf.shape(output)[0] // 2:, :])
                    else:
                        loss_value = loss_function(y_pred = output, y_true = targets)


                    #loss_value = loss_function(y_pred = output, y_true = targets)
                    loss_value += tf.nn.scale_regularization_loss (tf.reduce_sum(model.losses))

                gradients = g.gradient(loss_value, model.trainable_variables)

                if contrastive:
                    num_total_k_mer, num_correct_k_mer  =tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), tf.convert_to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
                    class_conc = tf.reshape(tf.convert_to_tensor([0., 0., 0.]), [1, 3])
                else:

                    num_total_k_mer, num_correct_k_mer = compute_k_mer_concordance(targets, output, data)

                    class_conc = compute_class_concordance(targets, output)


                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss_value, num_total_k_mer, num_correct_k_mer, class_conc



    if arguments['train']:
        try:
            resume_from = int(arguments["resume_from"])
            if resume_from < 1:
                saved_epochs = get_saved_epochs(train_directory)
                resume_from = saved_epochs[-1]
        except:
            resume_from = False

        if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
            n_train_samples_for_data = int(train_opts["n_samples"])
        else:
            n_train_samples_for_data = -1

        epochs = int(arguments["epochs"])
        save_interval = int(arguments["save_interval"])

        data = alt_data_generator(filebase= data_prefix,
                        batch_size = batch_size,
                        normalization_mode = norm_mode,
                        normalization_options = norm_opts,
                        impute_missing = fill_missing,
                        sparsifies  = sparsifies,
                        recombination_rate= recomb_rate,
                        generations = generations,
                        only_recomb=True,
                        contrastive = contrastive,
                        decode_only = decode_only,
                        n_samples = n_train_samples_for_data )

        chief_print("Recombination rate: " + str(data.recombination_rate))

        n_markers = data.n_markers
        if holdout_val_pop is None:
            data.define_validation_set2(validation_split= validation_split)
        else:
            data.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)


        data.missing_mask_input = missing_mask_input

        n_unique_train_samples = copy.deepcopy(data.n_train_samples)
        n_train_samples = n_unique_train_samples
        n_valid_samples = copy.deepcopy(data.n_valid_samples)


        if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
            n_train_samples = int(train_opts["n_samples"])
        else:
            n_train_samples = n_unique_train_samples

        batch_size_valid = batch_size
        n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size)
        _, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)
        data.n_train_samples_last_batch = int(n_train_samples_last_batch)
        data.n_valid_samples_last_batch = int(n_valid_samples_last_batch)

        data.n_train_samples = n_train_samples
        data.sample_idx_train = np.arange(n_train_samples)

        if holdout_val_pop is not None:
            if  data.recombination_rate != 0:

                data.create_temp_fam_and_superpop(data_prefix, superpopulations_file, n_train_batches)
                superpopulations_file = superpopulations_file+"_temp"

            #superpops = pd.read_csv(superpopulations_file, header=None).to_numpy()


        train_times = []
        train_epochs = []
        save_epochs = []

        ############### setup learning rate schedule ##############
        step_counter = resume_from * n_train_batches
        if "lr_scheme" in train_opts.keys():
            schedule_module = getattr(eval(train_opts["lr_scheme"]["module"]), train_opts["lr_scheme"]["class"])
            schedule_args = train_opts["lr_scheme"]["args"]

            if "decay_every" in schedule_args:
                decay_every = int(schedule_args.pop("decay_every"))
                decay_steps = n_train_batches * decay_every
                schedule_args["decay_steps"] = decay_steps

            if "t_mul" in schedule_args and resume_from:
                if schedule_args["t_mul"]  > 1:
                    n_restarts =  tf.math.floor( tf.math.log(1 + (schedule_args["t_mul"]-1 ) *step_counter / schedule_args["first_decay_steps"]  ) / tf.math.log(2.) )

                    schedule_args["first_decay_steps"] = schedule_args["first_decay_steps"] * (1- schedule_args["t_mul"]**(1 +n_restarts)) / (1- schedule_args["t_mul"]) - schedule_args["first_decay_steps"] * (1- schedule_args["t_mul"]**(n_restarts)) / (1- schedule_args["t_mul"])



            lr_schedule = schedule_module(learning_rate, **schedule_args)

            # use the schedule to calculate what the lr was at the epoch were resuming from
            updated_lr = lr_schedule(step_counter)
            #TODO  Here, also update the t_mul and m_mul for cosine decay. I.e., need to update the "first_decay_steps" to something new.

            lr_schedule = schedule_module(updated_lr, **schedule_args)

            chief_print("Using learning rate schedule {0}.{1} with {2}".format(train_opts["lr_scheme"]["module"], train_opts["lr_scheme"]["class"], schedule_args))
        else:
            lr_schedule = False

        chief_print("\n______________________________ Data ______________________________")
        chief_print("N unique train samples: {0}".format(n_unique_train_samples))
        chief_print("--- training on : {0}".format(n_train_samples))
        chief_print("N valid samples: {0}".format(n_valid_samples))
        chief_print("N markers: {0}".format(n_markers))
        chief_print("")


        chief_print("\n______________________________ Train ______________________________")
        chief_print("Model layers and dimensions:")
        chief_print("-----------------------------")

        chunk_size = 5 * data.batch_size
        data.encoded_things = tf.convert_to_tensor([np.tile(np.array([0,0]), (n_train_samples,1))])
        ds = data.create_dataset_tf_record(chunk_size, "training",n_workers = num_workers) #, device_id = device_id)

        if _isChief() and not contrastive:

            tf.print("Baseline concordance:", data.baseline_concordance)
            for i in range(5):
                tf.print("Baseline concordance {}-mer:".format(k_vec[i]), data.baseline_concordances_k_mer[i])


        def dataset_fn_train(input_context):
            #device_id = input_context.input_pipeline_id
            return data.create_dataset_tf_record(chunk_size, "training",n_workers = num_workers, shuffle=False) #, device_id = device_id)

        input_options = tf.distribute.InputOptions(
                experimental_place_dataset_on_device = False,
                experimental_fetch_to_device = True,
                experimental_replication_mode = tf.distribute.InputReplicationMode.PER_WORKER,
                experimental_per_replica_buffer_size = 1)

        dds = strategy.distribute_datasets_from_function(dataset_fn_train,input_options )
        if n_valid_samples > 0:
            ds_validation = data.create_dataset_tf_record(chunk_size, "validation",n_workers = num_workers) #, device_id = device_id)

            def dataset_fn_validation(input_context):
                #device_id = input_context.input_pipeline_id
                return data.create_dataset_tf_record(chunk_size, "validation", n_workers=num_workers,
                                                    shuffle=False)  # , device_id = device_id)


            dds_validation = strategy.distribute_datasets_from_function(dataset_fn_validation, input_options)

        with strategy.scope():
            # Initialize the model and optimizer
            autoencoder = Autoencoder(model_architecture, n_markers, noise_std, regularizer)
            autoencoder2 = None #  Autoencoder(model_architecture, n_markers, noise_std, regularizer)

            if pheno_model_architecture is not None:
                pheno_model = Autoencoder(pheno_model_architecture, 2, noise_std, regularizer)
            else:
                pheno_model = None

            if contrastive:
                optimizer = tf.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9,nesterov = True) # , beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)
            else:

                optimizer = tf.optimizers.Adam(learning_rate = lr_schedule, beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)
                #optimizer = tf.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9,nesterov = True) # , beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)
                #optimizer = tf.optimizers.Nadam(learning_rate = 0.01) # , beta_1=0.9, beta_2 = 0.999) # , amsgrad = True)

            optimizer2 = tf.optimizers.Adam(learning_rate = lr_schedule, beta_1=0.99, beta_2 = 0.999)

            input_test, _, _,_,_ = next(ds.as_numpy_iterator())
            _, _ = autoencoder(input_test[:,:,:], is_training = False, verbose = True)

          

            if resume_from:
                chief_print("\n______________________________ Resuming training from epoch {0} ______________________________".format(resume_from))
                weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", resume_from)
                chief_print("Reading weights from {0}".format(weights_file_prefix))
                autoencoder.load_weights(weights_file_prefix)

                input_test, _, _, _ ,_ = next(ds.as_numpy_iterator())
                _, _ = autoencoder(input_test[:,:,:], is_training = False, verbose = True)

        ######### Create objects for tensorboard summary ###############################
        #if isChief:
        #    train_writer = tf.summary.create_file_writer(train_directory + '/train')
        #    valid_writer = tf.summary.create_file_writer(train_directory + '/valid')
        ######################################################

        if _isChief():
            tf.print(autoencoder.summary())

            memory_info = tf.config.experimental.get_memory_info('GPU:0')

            #tf.print(memory_info["current"])
            #tf.print(memory_info["peak"])

        # train losses per epoch
        losses_t = []
        conc_t = []
        # valid losses per epoch
        losses_v = []
        conc_v = []

        compute_pop_baseline = False

        if isChief:
            if not os.path.isdir(train_directory + "/stats/"):
                os.mkdir(train_directory + "/stats/")
            if not contrastive:
                if os.path.isfile(train_directory + "/stats/baseline_concordances.csv"):
                    os.remove(train_directory + "/stats/baseline_concordances.csv")

                if compute_pop_baseline:
                    baselines = tf.stack(
                        [data.baseline_concordances_k_mer, tf.convert_to_tensor([data.baseline_concordance_superpop_informed]),
                        tf.convert_to_tensor([data.baseline_concordance_pop_informed])], axis=0).numpy()
                    with open(train_directory + "/stats/baseline_concordances.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in baselines])


        """"
        To enable the profiler, set profile to 1.
        
        Used for debugging the code / profiling different runs. Visualized in TensorBoard. I like to use ThinLinc to look at it, seems like the
        least effort solution to get tensorboard up and running. Possible to do it in other ways, but more painful.
        
        Another solution is to download the log files, and just run TensorBoard locally.
        """
        if "SLURM_PROCID" in os.environ:
            suffix = str(os.environ["SLURM_PROCID"])
        else:
            suffix = ""
        #if slurm_job:
        #	suffix = str(os.environ["SLURMD_NODENAME"])
        logs = train_directory+ "/logdir/"  + datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+ suffix
        samples_seen = 0

        profile = 0

        for batch_dist_input, batch_dist_target, poplist, _ , _ in ds :
            samples_seen += tf.shape(batch_dist_input)[0]


        if "SLURM_PROCID" in os.environ:
            print(" samples seen on process " + str(os.environ["SLURM_PROCID"]) + ":  " + str(samples_seen))
        else:
            print("Samples seen: " + str(samples_seen))




        for e in range(1,epochs+1):

            # The dataset needs to be updated if we want to use past epoch coordinates in training, this may cause slow down, so skip it if not needed
            # Update: Want to enable this for contrastive learning, where we want to shuffle the files that the workers read so that each worker does not always see the same samples.

            need_encoded_things= 0

            if need_encoded_things or data.contrastive:
                dds = strategy.distribute_datasets_from_function(dataset_fn_train,input_options )


            if e % 100 < 50:
                autoencoder.passthrough.assign(0.0)
            else:
                autoencoder.passthrough.assign(1.0)

            if e % 200 < 100:
                autoencoder.cell_lock.assign(1.0)
            else:
                autoencoder.cell_lock.assign(-5)
            if e ==2:
                if profile : tf.profiler.experimental.start(logs)

            startTime = datetime.now()
            effective_epoch = e + resume_from
            train_loss = 0
            conc_train_total_k_mer  = 0
            conc_train_correct_k_mer = 0
            class_conc = tf.reshape(tf.convert_to_tensor([0.,0.,0.]),[1,3])
            class_conc_valid= tf.reshape(tf.convert_to_tensor([0.,0.,0.]),[1,3])
            raw_encoding = None


            for batch_dist_input, batch_dist_target, poplist, original_genotypes, original_inds in dds:
                pt = generatepheno(phenodata, poplist)

                train_batch_loss, num_total_k_mer,num_correct_k_mer, class_conc_temp= distributed_train_step(autoencoder, autoencoder2, optimizer, optimizer2, loss_func, batch_dist_input, batch_dist_target, False, phenomodel=pheno_model, phenotargets=pt,data=data)
                conc_train_total_k_mer += num_total_k_mer
                conc_train_correct_k_mer += num_correct_k_mer

                if need_encoded_things:
                    raw_encoded_data = autoencoder(original_genotypes, encode_only=True,is_training= False)
                    if raw_encoding == None:
                        raw_encoding = raw_encoded_data
                    else:
                        raw_encoding = tf.concat([raw_encoding, raw_encoded_data], axis=0)

                        data.encoded_things = raw_encoding
                """
                This snippet verifies that no held-out samples get passed to training
                if holdout_val_pop is not None and num_devices == 1 and num_workers == 1 :

                    sample_superpop = np.array([superpops[np.where(poplist[i,1] == superpops[:, 0])[0][0], 1] for i in range(len(poplist[:,1]))])
                    assert(np.sum(np.where(sample_superpop == "holdout_val_pop")[0]) == 0 )
                """

                #print(tf.shape(strategy.experimental_local_results(poplist)))
                train_loss += train_batch_loss
                class_conc += class_conc_temp
            if data.only_recomb:
                n_train_samples = 0
                chief_print("Observe: only passing recombined samples, no original samples. ")

            train_loss_this_epoch = train_loss /  (n_train_samples  + n_train_batches* (data.total_batch_size -data.batch_size))
            if not contrastive:
                train_conc_this_epoch = conc_train_correct_k_mer[0] / conc_train_total_k_mer[0]
                train_conc_this_epoch_k_mer = conc_train_correct_k_mer / conc_train_total_k_mer
            else:
                train_conc_this_epoch = 0
                train_conc_this_epoch_k_mer = tf.convert_to_tensor(tf.zeros([1,len(k_vec)]))
            class_conc_this_epoch = tf.cast(class_conc,tf.float32) / tf.cast(tf.reduce_sum(A,axis = 0),tf.float32) # n_markers/data.n_train_samples

            train_time = (datetime.now() - startTime).total_seconds()
            train_times.append(train_time)
            train_epochs.append(effective_epoch)
            losses_t.append(train_loss_this_epoch)
            conc_t.append(train_conc_this_epoch)


            if not contrastive:
                #chief_print(class_conc_this_epoch)
                pass


            train_error_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), tf.reshape(train_loss_this_epoch,[1,1])],axis = 1).numpy()[0]
            if not os.path.isdir(train_directory+"/stats/"):
                os.mkdir(train_directory+"/stats/")

            if isChief:
                with open(train_directory + "/stats/training_loss.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([str(i) for i in train_error_write])
                lr_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), tf.reshape(lr_schedule(optimizer.iterations),[1,1])],axis = 1).numpy()[0]

                with open(train_directory + "/stats/learning_rate.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([str(i) for i in lr_write])
                if not contrastive:
                    train_conc_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), tf.reshape(train_conc_this_epoch_k_mer,[1,5])],axis = 1).numpy()[0]
                    with open(train_directory + "/stats/training_k_mer_concordance.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in train_conc_write])

                    class_conc_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), class_conc_this_epoch],axis = 1).numpy()[0]


                    with open(train_directory + "/stats/training_class_concordance.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in class_conc_write])

            if e == 1:
                conc_t_k_mer = tf.reshape(train_conc_this_epoch_k_mer, [1, 5])
                class_conc_tot = tf.reshape(class_conc_this_epoch,[1,3])
            else:
                conc_t_k_mer = tf.concat([conc_t_k_mer, tf.reshape(train_conc_this_epoch_k_mer, [1, 5])], axis=0)
                class_conc_tot = tf.concat([class_conc_tot, tf.reshape(class_conc_this_epoch, [1, 3])], axis=0)

            chief_print("Epoch: {}/{}...".format(effective_epoch, epochs+resume_from))
            chief_print("--- Train loss: {:.4f} Concordance {:.4f} time: {}".format(train_loss_this_epoch, train_conc_this_epoch_k_mer[0,0],  train_time))
            #chief_print("Learning rate: {}".format(lr_schedule(optimizer.iterations)))
            if n_valid_samples > 0:

                startTime = datetime.now()
                valid_loss = 0
                conc_valid_total_k_mer = 0
                conc_valid_correct_k_mer = 0
                for input_valid_batch, targets_valid_batch, poplist,original_genotypes,original_inds  in dds_validation:

                    valid_loss_batch, num_total_k_mer, num_correct_k_mer,class_conc_temp_valid = distributed_valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch,data)

                    valid_loss += valid_loss_batch

                    conc_valid_total_k_mer += num_total_k_mer
                    conc_valid_correct_k_mer += num_correct_k_mer
                    class_conc_valid += class_conc_temp_valid

                    """
                    This snippet verifies that no non-heldout sample gets passed to validation. This is for verification of holdout val-pop,
                    not intended to be run all the time
                    
                    if holdout_val_pop is not None and num_devices == 1 and num_workers == 1 :
                        sample_superpop = np.array([superpops[np.where(poplist[i, 1] == superpops[:, 0])[0][0], 1] for i in range(len(poplist[:, 1]))])

                        assert (np.sum(np.where(sample_superpop != holdout_val_pop)[0]) == 0)
                    """
                #valid_conc_this_epoch = conc_valid_correct / conc_valid_total
                # Instead of having a separate one for the k=1 case, just use the one from the k-vector
                if not contrastive:
                    valid_conc_this_epoch = conc_valid_correct_k_mer[0] / conc_valid_total_k_mer[0]
                    valid_conc_this_epoch_k_mer = conc_valid_correct_k_mer / conc_valid_total_k_mer
                else:
                    valid_conc_this_epoch = 0
                    valid_conc_this_epoch_k_mer = tf.convert_to_tensor(tf.zeros([1, len(k_vec)]))


                class_conc_this_epoch = tf.cast(class_conc,tf.float32) / tf.cast(tf.reduce_sum(A,axis = 0),tf.float32) # n_markers/data.n_train_samples


                valid_loss_this_epoch = valid_loss  / n_valid_samples
                class_conc_valid_this_epoch = tf.cast(class_conc_valid,tf.float32) / tf.cast(tf.reduce_sum(A_valid,axis = 0),tf.float32) # n_markers/data.n_train_samples

                valid_error_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), tf.reshape(valid_loss_this_epoch,[1,1])],axis = 1).numpy()[0]
                if isChief:
                    with open(train_directory + "/stats/validation_loss.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in valid_error_write])

                valid_conc_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), tf.reshape(valid_conc_this_epoch_k_mer,[1,5])],axis = 1).numpy()[0]
                if isChief:
                    with open(train_directory + "/stats/validation_k_mer_concordance.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in valid_conc_write])

                class_conc_write = tf.concat([tf.reshape(tf.convert_to_tensor(tf.cast(effective_epoch,tf.float32)), [1, 1]), class_conc_valid_this_epoch],axis = 1).numpy()[0]
                if isChief:
                    with open(train_directory + "/stats/validation_class_concordance.csv", 'a', newline='') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([str(i) for i in class_conc_write])

                if e == 1:
                    class_conc_valid_tot = tf.reshape(class_conc_valid_this_epoch,[1,3])
                else:
                    class_conc_valid_tot = tf.concat([class_conc_valid_tot, tf.reshape(class_conc_valid_this_epoch, [1, 3])], axis=0)

                losses_v.append(valid_loss_this_epoch)
                conc_v.append(valid_conc_this_epoch)
                if e == 1:
                    conc_v_k_mer = tf.reshape(valid_conc_this_epoch_k_mer, [1,5])
                else:
                    conc_v_k_mer = tf.concat([conc_v_k_mer, tf.reshape(valid_conc_this_epoch_k_mer, [1,5])], axis = 0)



                valid_time = (datetime.now() - startTime).total_seconds()
                chief_print("--- Valid loss: {:.4f} Concordance {:.4f} time: {}".format(valid_loss_this_epoch, valid_conc_this_epoch_k_mer[0,0], valid_time))
            if _isChief():
                weights_file_prefix = train_directory + "/weights/" + str(effective_epoch)
            else:
                #weights_file_prefix = train_directory + "/weights/temp"+ str(os.environ["SLURM_PROCID"]) +"/" + str(effective_epoch)
                weights_file_prefix ="/scratch/local/"+ str(effective_epoch)+os.environ["SLURM_PROCID"] # Save to some junk directory, /scratch/local on Berra is a temp directory that deletes files after job is done.
            pheno_weights_file_prefix = train_directory + "/pheno_weights/" + str(effective_epoch)
            if e % save_interval == 0:
                startTime = datetime.now()
                save_weights(weights_file_prefix, autoencoder)
                #save_weights(pheno_weights_file_prefix, pheno_model)
                save_time = (datetime.now() - startTime).total_seconds()
                save_epochs.append(effective_epoch)
                chief_print("-------- Save time: {0} dir: {1}".format(save_time, weights_file_prefix))
        if profile: tf.profiler.experimental.stop()

        if isChief:
            outfilename = train_directory + "/" + "train_times.csv"
            write_metric_per_epoch_to_csv(outfilename, train_times, train_epochs)

            outfilename = "{0}/losses_from_train_t.csv".format(train_directory)
            #epochs_t_combined, losses_t_combined = write_metric_per_epoch_to_csv(outfilename, losses_t, train_epochs)
            _, ax = plt.subplots()
            loss_train =pd.read_csv(train_directory + "/stats/training_loss.csv",header = None).to_numpy()
            plt.plot(loss_train[:,0], loss_train[:,1], label="train", c="orange")

            #plt.plot(epochs_t_combined, losses_t_combined, label="train", c="orange")

            if n_valid_samples > 0:
                outfilename = "{0}/losses_from_train_v.csv".format(train_directory)
                #epochs_v_combined, losses_v_combined = write_metric_per_epoch_to_csv(outfilename, losses_v, train_epochs)

                loss_valid = pd.read_csv(train_directory + "/stats/validation_loss.csv", header=None).to_numpy()
                #plt.plot(epochs_v_combined, losses_v_combined, label="valid", c="blue")
                plt.plot(loss_valid[:,0], loss_valid[:,1], label="valid", c="blue")

                min_valid_loss_epoch = loss_valid[np.argmin(loss_valid[:,1]),0]
                min_valid_loss = np.min(loss_valid[:,1])
                plt.axvline(min_valid_loss_epoch, color="black")
                plt.text(min_valid_loss_epoch + 0.1, 0.5,'min valid loss at epoch {}'.format(int(min_valid_loss_epoch)),
                        rotation=90,
                        transform=ax.get_xaxis_text1_transform(0)[0])
                plt.title(" Min Valid loss: {:.4f}".format(min_valid_loss))
            plt.xlabel("Epoch")
            plt.ylabel("Loss function value")
            plt.legend()
            plt.savefig("{}/stats/losses_from_train.pdf".format(train_directory))
            plt.close()

            if not contrastive:
                plt.figure()

                ct = pd.read_csv(train_directory + "/stats/training_k_mer_concordance.csv",header= None).to_numpy()
                cv = pd.read_csv(train_directory + "/stats/validation_k_mer_concordance.csv",header= None).to_numpy()

                plt.plot(ct[:,0],ct[:,1], label = "Training", linewidth = 2)
                plt.plot(cv[:,0],cv[:,1], label = "Validation",  linewidth = 2)

                plt.plot(np.ones(len(np.arange(1,np.max(cv[:,0])))) * data.baseline_concordance, 'k',  linewidth = 2, label = "Baseline")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Genotype Concordance")
                plt.savefig("{}/stats/concordances.pdf".format(train_directory))
                plt.close()

                plt.figure()
                for i in range(conc_v_k_mer.shape[1]):
                    plt.plot(cv[:, i+1],color = "C{}".format(i), label="Validation {}-mer".format(k_vec[i]), linewidth=2)
                    plt.plot(np.ones(len(conc_t)) * data.baseline_concordances_k_mer[i], linestyle='dashed',color = "C{}".format(i), linewidth=2, label="Baseline{}-mer".format(k_vec[i]))
                    plt.plot(cv[:, 0], np.ones(len(cv[:, 0])) * data.baseline_concordances_k_mer[i], linestyle='dashed',
                            color="C{}".format(i), linewidth=2, label="Baseline{}-mer".format(k_vec[i]))


                plt.legend(prop ={"size":3} )
                plt.xlabel("Epoch")
                plt.ylabel("Genotype Concordance")
                plt.savefig("{}/stats/concordances_k_meres.pdf".format(train_directory))
                plt.close()
                plt.figure()
                for i in range(cv.shape[1]-1):
                    plt.plot(cv[:, 0],cv[:, i+1], color="C{}".format(i), label="Validation {}-mer".format(k_vec[i]),
                            linewidth=2)
                    plt.plot(cv[:, 0],ct[:, i+1], linestyle='dotted',color="C{}".format(i), label="training {}-mer".format(k_vec[i]),
                            linewidth=2)
                    plt.plot(cv[:, 0],np.ones(len(cv[:, 0])) * data.baseline_concordances_k_mer[i], linestyle='dashed',
                            color="C{}".format(i), linewidth=2, label="Baseline{}-mer".format(k_vec[i]))

                plt.legend(prop={"size": 3})
                plt.xlabel("Epoch")
                plt.ylabel("Genotype Concordance")
                plt.savefig("{}/stats/concordances_both_k_meres.pdf".format(train_directory))
                plt.close()

                chief_print(conc_v_k_mer.numpy()[-1,:]/ data.baseline_concordances_k_mer)

                plt.figure(figsize = (10,5))

                class_conc_tot =pd.read_csv(train_directory + "/stats/training_class_concordance.csv",header = None).to_numpy()

                plt.subplot(1,2,1)
                plt.plot(class_conc_tot[:,1],label = "0", linewidth=2)

                plt.plot(class_conc_tot[:,2],label = "1", linewidth=2)
                plt.plot(class_conc_tot[:,3],label = "2", linewidth=2)

                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("class concordance")

                plt.subplot(1,2,2)
                conc_partition_by_class =    tf.cast(class_conc_tot[:,1:],tf.float32) *  tf.cast(tf.reduce_sum(A,axis = 0),tf.float32) / tf.cast(n_markers,tf.float32) / tf.cast((n_train_samples  + n_train_batches* (data.total_batch_size -data.batch_size)),tf.float32)

                plt.plot(conc_partition_by_class.numpy()[:,0],label = "0" ,linewidth=2)
                plt.plot(conc_partition_by_class.numpy()[:,1],label = "1", linewidth=2)
                plt.plot(conc_partition_by_class.numpy()[:,2],label = "2", linewidth=2)

                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Total concordance, partioned by class")
                plt.savefig("{}/stats/class_concordances_t.pdf".format(train_directory))
                plt.close()

                plt.figure(figsize = (10,5))

                class_conc_valid_tot =pd.read_csv(train_directory + "/stats/validation_class_concordance.csv",header = None).to_numpy()

                plt.subplot(1,2,1)
                plt.plot(class_conc_valid_tot[:,1],label = "0", linewidth=2)

                plt.plot(class_conc_valid_tot[:,2],label = "1", linewidth=2)
                plt.plot(class_conc_valid_tot[:,3],label = "2", linewidth=2)

                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("class concordance")

                plt.subplot(1,2,2)
                conc_partition_by_class_valid=    tf.cast(class_conc_valid_tot[:,1:],tf.float32) *  tf.cast(tf.reduce_sum(A_valid,axis = 0),tf.float32) / tf.cast(n_markers,tf.float32) / tf.cast(n_valid_samples ,tf.float32)
                plt.plot(conc_partition_by_class_valid.numpy()[:,0],label = "0" ,linewidth=2)
                plt.plot(conc_partition_by_class_valid.numpy()[:,1],label = "1", linewidth=2)
                plt.plot(conc_partition_by_class_valid.numpy()[:,2],label = "2", linewidth=2)

                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Total concordance, partioned by class")
                plt.savefig("{}/stats/class_concordances_v.pdf".format(train_directory))
                plt.close()


        chief_print("Done training. Wrote to {0}".format(train_directory))

    if arguments['project']:

        projected_epochs = get_projected_epochs(encoded_data_file)

        if arguments['epoch']:
            epoch = int(arguments['epoch'])
            epochs = [epoch]

        else:
            epochs = get_saved_epochs(train_directory)

        for projected_epoch in projected_epochs:
            try:
                epochs.remove(projected_epoch)
            except:
                continue

        chief_print("Projecting epochs: {0}".format(epochs))
        chief_print("Already projected: {0}".format(projected_epochs))

        # Make this larger. For some reason a low project batch size resulted in division by zero in the normalization, which yields a bad scaler
        batch_size_project = int(batch_size)
        sparsify_fraction = 0.0

        if alt_data is not None:
            data_prefix = datadir+alt_data

        data = alt_data_generator(filebase= data_prefix,
                batch_size = batch_size_project,
                normalization_mode = norm_mode,
                normalization_options = norm_opts,
                impute_missing = fill_missing,
                recombination_rate = recomb_rate,
                generations = generations,
                only_recomb=False,
                contrastive = False ,# contrastive,
                n_samples = n_train_samples)
        data.sparsifies = [sparsify_fraction]
        data._define_samples() # This sets the number of validation samples to be 0
        ind_pop_list_train_reference = data.ind_pop_list_train_orig[data.sample_idx_train]

        write_h5(encoded_data_file, "ind_pop_list_train", np.array(ind_pop_list_train_reference, dtype='S'))

        n_train_samples = copy.deepcopy(data.n_train_samples)

        n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size_project)
        n_valid_samples = 0

        batch_size_valid = 1
        _, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)

        data.n_valid_samples_last_batch = n_valid_samples_last_batch
        data.n_train_samples_last_batch = n_train_samples_last_batch

        data.missing_mask_input = missing_mask_input

        # If we want to also plot recombined samples in the projected plots, create and use temporary fam and superpop
        # files. This will include recombined samples in the plots and legend in a better way than what I had done before.
        if data.recombination_rate != 0:

            data.create_temp_fam_and_superpop(data_prefix, superpopulations_file, n_train_batches)
            superpopulations_file = superpopulations_file+"_temp"
            proj_fam_file = data_prefix+"_temp"

        else:
            proj_fam_file = data_prefix

        ############################
        # Here, create new dataset, with the same train split as in the training step.
        # This is for plotting/ showing which of the samples have been used as training/validation data. It will become the same since I use a fixed seed inside data_handler.

        # Here, I think we need to have recombinnation rate either 0 or 1, otherwise
        data_project = alt_data_generator(filebase= data_prefix,
                        batch_size = batch_size,
                        normalization_mode = norm_mode,
                        normalization_options = norm_opts,
                        impute_missing = fill_missing,
                        sparsifies  = sparsifies,
                        only_recomb = True)
        if recomb_rate != 0:
            if recomb_rate !=1:

                print("set recomb rate to either 0 or 1, need to fix this later.")
        n_markers = data_project.n_markers
        data_project.sparsifies = [sparsify_fraction]

        if holdout_val_pop is None:
            data_project.define_validation_set2(validation_split= validation_split)
        else:
            data_project.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)

        data_project.missing_mask_input = missing_mask_input

        n_unique_train_samples = copy.deepcopy(data_project.n_train_samples)
        n_valid_samples = copy.deepcopy(data_project.n_valid_samples)


        if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
            n_train_samples = int(train_opts["n_samples"])
        else:
            n_train_samples = n_unique_train_samples

        batch_size_valid = batch_size
        n_train_batches, n_train_samples_last_batch  = get_batches(n_train_samples, batch_size)
        _, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)

        data_project.n_train_samples_last_batch = int(n_train_samples_last_batch)
        data_project.n_valid_samples_last_batch = int(n_valid_samples_last_batch)


        chunk_size = 5 * data_project.batch_size


        # loss function of the train set per epoch
        losses_train = []

        # genotype concordance of the train set per epoch
        genotype_concs_train = []
        genotype_concordance_metric = GenotypeConcordance()

        autoencoder = Autoencoder(model_architecture, n_markers, 0, regularizer)
        if pheno_model_architecture is not None:
            pheno_model = Autoencoder(pheno_model_architecture, 2, noise_std, regularizer)
        else:
            pheno_model = None
            pheno_train = None


        genotype_concordance_metric = GenotypeConcordance()

        scatter_points_per_epoch = []
        colors_per_epoch = []
        markers_per_epoch = []
        edgecolors_per_epoch = []

        data.batch_size = batch_size_project
        chunk_size = 5 * data.batch_size
        pheno_train = None
     

        data.encoded_things = tf.convert_to_tensor([np.tile(np.array([0,0]), (2067,1))])

        ds = data.create_dataset_tf_record(chunk_size, "project",n_workers=num_workers)

        def dataset_fn_project(input_context):

            return data.create_dataset_tf_record(chunk_size, "project", n_workers=num_workers,
                                                    shuffle=True)

        input_options = tf.distribute.InputOptions(
            experimental_place_dataset_on_device=False,
            experimental_fetch_to_device=True,
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
            experimental_per_replica_buffer_size=1)

        dds = strategy.distribute_datasets_from_function(dataset_fn_project, input_options)

        autoencoder.noise_std = 0

        poplist = pd.read_csv(proj_fam_file+".fam", header=None).to_numpy()
        poplist2 = np.empty([poplist.shape[0], 6]).astype(str)
        for i in np.arange(poplist.shape[0]):
            poplist2[i, :] = np.array(poplist[i][0].split(" "))  # str(poplist[i][0].split(" "))


        def compute_and_save_binned_conc(pred,true,name):
            #TODO: This has an error right now that I have not had time to check. Do not use 
            chief_print("WARNING: Using function \"compute_and_save_binned_conc\", this has an unfixed error and should not be used. ")
            c0 = np.sum(((pred-true==0) * (true==0)).astype(int),axis = 0) # How many are predicted correctly, and are genotype 0, per SNP
            c1 = np.sum(((pred-true==0) * (true==1)).astype(int),axis = 0) # How many are predicted correctly, and are genotype 1, per SNP
            c2 = np.sum(((pred-true==0) * (true==2)).astype(int),axis = 0) # How many are predicted correctly, and are genotype 2, per SNP

            d0 = np.sum((true == 0).astype(int), axis = 0)
            d1 = np.sum((true == 1).astype(int), axis = 0)
            d2 = np.sum((true == 2).astype(int), axis = 0)

            d2[d2==0] = 1
            d1[d1==0] = 1
            d0[d0==0] = 1

            binned_conc0 = np.zeros(len(bins))
            binned_conc1 = np.zeros(len(bins))
            binned_conc2 = np.zeros(len(bins))

            for bin_nr in range(len(bins)-1):
                indices = np.where(bin_inds == bin_nr)

                binned_conc0[bin_nr] = np.sum(c0[indices]) / sum(d0[indices])
                binned_conc1[bin_nr] = np.sum(c1[indices]) / sum(d1[indices])
                binned_conc2[bin_nr] = np.sum(c2[indices]) / sum(d2[indices])

            binned_conc_this_epoch0 = tf.concat([ tf.reshape(tf.convert_to_tensor([tf.cast(epoch,tf.float32)]), [1,1]), tf.reshape(tf.cast(tf.convert_to_tensor(binned_conc0),tf.float32), [1,len(bins)])], axis = 1)
            binned_conc_this_epoch1 = tf.concat([ tf.reshape(tf.convert_to_tensor([tf.cast(epoch,tf.float32)]), [1,1]), tf.reshape(tf.cast(tf.convert_to_tensor(binned_conc1),tf.float32), [1,len(bins)])], axis = 1)
            binned_conc_this_epoch2 = tf.concat([ tf.reshape(tf.convert_to_tensor([tf.cast(epoch,tf.float32)]), [1,1]), tf.reshape(tf.cast(tf.convert_to_tensor(binned_conc2),tf.float32), [1,len(bins)])], axis = 1)

            if isChief:
                with open(results_directory + "/" + name+ "0.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([str(i) for i in binned_conc_this_epoch0.numpy()[0,:]])
            if isChief:
                with open(results_directory + "/"  + name+ "1.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([str(i) for i in binned_conc_this_epoch1.numpy()[0,:]])
            if isChief:
                with open(results_directory + "/" + name+ "2.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([str(i) for i in binned_conc_this_epoch2.numpy()[0,:]])



            return binned_conc0,binned_conc1,binned_conc2


        for epoch in epochs:
            #dds = strategy.distribute_datasets_from_function(dataset_fn, input_options)

            t1 = time.perf_counter()
            chief_print("########################### epoch {0} ###########################".format(epoch))
            weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", epoch)
            chief_print("Reading weights from {0}".format(weights_file_prefix))

            autoencoder.load_weights(weights_file_prefix)
            if pheno_model is not None:
                pheno_weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "pheno_weights", epoch)
                pheno_model.load_weights(pheno_weights_file_prefix)


            ind_pop_list_train = np.empty((0,2))
            encoded_train = np.empty((0, n_latent_dim))
            encoded_train_recombined = np.empty((0, n_latent_dim))

            encoded_train_to_write = np.empty((0, n_latent_dim))

            decoded_train = None
            targets_train = np.empty((0, n_markers))

            loss_value_per_train_batch = []
            loss_train_batch = 0
            pop_inds = np.empty((0, 1))

            raw_encoding = None

            for input_train_batch, targets_train_batch, ind_pop_list_train_batch,original_genotypes, original_inds in dds:

                if data.recombination_rate != 0:
                    input_train_batch = tf.concat([original_genotypes, input_train_batch], axis=0)
                    targets_train_batch = tf.concat([original_genotypes[:,:,0], targets_train_batch], axis = 0)
                    ind_pop_list_train_batch = tf.concat([original_inds,ind_pop_list_train_batch ], axis = 0)

                loss_train_batch, num_total_k_mer, num_correct_k_mer,class_conc_temp_valid, decoded_train_batch,encoded_train_batch, pop_inds_batch,targets_train_batch= distributed_proj_batch(autoencoder, loss_func, input_train_batch, targets_train_batch,data,poplist2,ind_pop_list_train_batch)


                # Here pop_inds are linking coords with the index in the fam file, excluding the recombined samples
                tmp = pop_inds_batch.numpy()[:,np.newaxis]
                pop_inds = np.concatenate((pop_inds,tmp),axis = 0).astype(int)
                if data.recombination_rate != 0:
                    encoded_train = np.concatenate((encoded_train, encoded_train_batch[0:tf.cast(tf.shape(encoded_train_batch)[0]/2,tf.int32),:]), axis=0)
                    encoded_train_recombined = np.concatenate((encoded_train_recombined, encoded_train_batch[tf.cast(tf.shape(encoded_train_batch)[0]/2,tf.int32):,:]), axis=0)
                else:
                    encoded_train = np.concatenate((encoded_train, encoded_train_batch[0:tf.cast(tf.shape(encoded_train_batch)[0],tf.int32),:]), axis=0)
                    encoded_train_recombined = np.concatenate((encoded_train_recombined, encoded_train_batch[tf.cast(tf.shape(encoded_train_batch)[0],tf.int32):,:]), axis=0)

                # Since we want to plot the recombined samples, we save the original ones separately for writing to the encoded  h5 file used in evaluate.
                # This may have to be done differently in the distributed setting
                encoded_train_to_write = np.concatenate((encoded_train_to_write, encoded_train_batch[:-data.total_batch_size + data.batch_size,:]), axis=0)

                if decoded_train is None:
                    decoded_train = np.copy(decoded_train_batch[:,0:n_markers])
                else:
                    decoded_train = np.concatenate((decoded_train, decoded_train_batch[:,0:n_markers]), axis=0)

                targets_train = np.concatenate((targets_train, targets_train_batch[:,0:n_markers]), axis=0)
                loss_value_per_train_batch.append(loss_train_batch)

                need_encoded = 0 # most likely will not need it
                if need_encoded:
                    raw_encoded_data = autoencoder(tf.convert_to_tensor(original_genotypes), encode_only=True)
                    if raw_encoding == None:
                        raw_encoding = raw_encoded_data
                    else:
                        raw_encoding = tf.concat([raw_encoding, raw_encoded_data], axis=0)

                    data.encoded_things = raw_encoding

            #TODO: Also stratify by train_test?  should be able to do, will take some coding though. I have the train indices...
            # Since I now write it so a csv, I do no longer really need to collect to a larger array. Can just write now -read later-
            # This also helps in multi-project settings

            ind_pop_list_train = poplist2[np.squeeze(pop_inds),:]
            encoded_train = np.array(encoded_train)
            encoded_train_to_write = np.array(encoded_train_to_write)


            if not contrastive and False:
                pred = 2 *(tf.cast(tf.argmax(alfreqvector(decoded_train[np.squeeze(pop_inds), 0:n_markers]), axis = -1), tf.float32) * 0.5).numpy()
                true = 2* targets_train[np.squeeze(pop_inds),:]

                compute_and_save_binned_conc(pred,true,name ="binned_concordances" )
                compute_and_save_binned_conc(pred[data_project.sample_idx_train,:],true[data_project.sample_idx_train,:],name ="binned_concordances_train" )
                if n_valid_samples != 0:
                    compute_and_save_binned_conc(pred[data_project.sample_idx_valid,:],true[data_project.sample_idx_valid,:],name ="binned_concordances_valid" )

            output_names = ind_pop_list_train[:,1]
            ind_pop_list_train2 = ind_pop_list_train[:,[1,0]]


            encoded_train_get_coords = tf.concat([encoded_train,encoded_train_recombined], axis = 0)
            ind_pop_list_train2 = tf.concat([ind_pop_list_train2,tf.tile(tf.convert_to_tensor(["zArtificial","Artificial_1x"])[tf.newaxis, :], [tf.shape(encoded_train_recombined)[0], 1])],axis = 0).numpy()
            ids = poplist2[:, 1]
            list_t = []
            list_v  = []
            list_all = []

            # Find train, validation, and all non-recombined samples.
            for i in ids[data_project.sample_idx_train]:
                x = np.where(output_names == i)[0]
                if len(x) >0:
                    list_t.append(x[0])

            try:

                for i in ids[data_project.sample_idx_valid]:
                    x = np.where(output_names == i)[0]
                    if len(x) >0:
                        list_v.append(x[0])
            except:
                list_v = []
            for i in ids[data_project.sample_idx_all]:
                x = np.where(output_names == i)[0]
                if len(x) >0:
                    list_all.append(x[0])

            encoded_train_to_write = encoded_train[np.array(list_all)]

            if isChief:

                list(ind_pop_list_train[:,1])
                list(ind_pop_list_train_reference[:,1])
                loss_value = np.sum(loss_value_per_train_batch)/n_train_samples # /num_devices

                if not fill_missing:
                    orig_nonmissing_mask = get_originally_nonmissing_mask(targets_train)
                else:
                    orig_nonmissing_mask = np.full(targets_train.shape, True)

                if train_opts["loss"]["class"] == "MeanSquaredError" and (data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):
                    try:
                        _ = data.scaler
                    except:
                        chief_print("Could not calculate predicted genotypes and genotype concordance. No scaler available in data handler.")
                        genotypes_output = np.array([])
                        true_genotypes = np.array([])

                    genotypes_output = to_genotypes_invscale_round(decoded_train[:, 0:n_markers], scaler_vals = [data.scaler.mean_, data.scaler.var_])
                    true_genotypes = to_genotypes_invscale_round(targets_train, scaler_vals = [data.scaler.mean_, data.scaler.var_])
                    genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask],
                                                            y_true = true_genotypes[orig_nonmissing_mask])


                elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
                    genotypes_output = to_genotypes_sigmoid_round(decoded_train[:, 0:n_markers])
                    true_genotypes = targets_train
                    genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask], y_true = true_genotypes[orig_nonmissing_mask])

                elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts["norm_mode"] == "genotypewise01" and False:
                    if alt_data is None:
                        genotypes_output = tf.cast(tf.argmax(alfreqvector(decoded_train[:, 0:n_markers]), axis = -1), tf.float32) * 0.5
                        true_genotypes = targets_train

                        genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask], y_true = true_genotypes[orig_nonmissing_mask])

                else:
                    chief_print("Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(train_opts["loss"]["class"],
                                                                                                                                                        data_opts["norm_mode"]))
                    genotypes_output = np.array([])
                    true_genotypes = np.array([])

                genotype_concordance_value = genotype_concordance_metric.result()

                losses_train.append(loss_value)
                genotype_concs_train.append(genotype_concordance_value)
                epoch_for_filename = "0"*(7 -len(str(epoch))) + str(epoch)

                if superpopulations_file:
                    #coords_by_pop = get_coords_by_pop(proj_fam_file, encoded_train)
                    if doing_clustering:
                        coords_by_pop = get_coords_by_pop(proj_fam_file,encoded_train_get_coords, ind_pop_list = ind_pop_list_train2)
                        t0 = time.perf_counter()
                        plot_clusters_by_superpop(coords_by_pop, "{0}/clusters_e_{1}".format(results_directory, epoch_for_filename), superpopulations_file, write_legend = epoch == epochs[0])
                        tf.print("Time for plotting clusters: ", time.perf_counter() - t0 )
                    #else:
                    if "box_area" in autoencoder.regularizer:
                        box_area = autoencoder.regularizer["box_area"]
                    else:
                        box_area = 0
                    coords_by_pop = get_coords_by_pop(proj_fam_file, encoded_train_get_coords, ind_pop_list = ind_pop_list_train2)

                    scatter_points, colors, markers, edgecolors = \
                        plot_coords_by_superpop(coords_by_pop,"{0}/dimred_by_superpop_e_{1}".format(results_directory, epoch_for_filename), superpopulations_file, plot_legend = epoch == epochs[0],epoch = str(epoch), box_area = box_area)

                    scatter_points_per_epoch.append(scatter_points)
                    colors_per_epoch.append(colors)
                    markers_per_epoch.append(markers)
                    edgecolors_per_epoch.append(edgecolors)

                else:
                    try:
                        coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
                        plot_coords_by_pop(coords_by_pop, "{0}/dimred_by_pop_e_{1}".format(results_directory, epoch_for_filename))
                    except:
                        plot_coords(encoded_train, "{0}/dimred_e_{1}".format(results_directory, epoch_for_filename))

                if pheno_train is not None:
                    writephenos(f'{results_directory}/{epoch}.phe', ind_pop_list_train, pheno_train)
                write_h5(encoded_data_file, "{0}_encoded_train".format(epoch), encoded_train_to_write)


                markersize = 5
                encoded_train2 = np.array(encoded_train)

                plt_kde=False
                if plt_kde:

                    train_test_KDE_plot(outfile_prefix="{0}/dimred_by_train_test_KDEtest_e_{1}".format(results_directory, epoch_for_filename),
                                        x_train=encoded_train2[np.array(list_t), 0],
                                        y_train=encoded_train2[np.array(list_t), 1],
                                        x_valid=encoded_train2[np.array(list_v), 0],
                                        y_valid=encoded_train2[np.array(list_v), 1],
                                        markersize=markersize)

            chief_print("Projection time for {} gpus: {} seconds".format(num_devices, time.perf_counter()- t1))


        if isChief and not contrastive:
            def plot_binned_conc(name, save_name):
                plt.figure(figsize = (15,5))
                binned_concordances0 = pd.read_csv(results_directory + "/" + name +"0.csv",header = None).to_numpy()
                binned_concordances1 = pd.read_csv(results_directory + "/" + name +"1.csv",header = None).to_numpy()
                binned_concordances2 = pd.read_csv(results_directory + "/" + name +"2.csv",header = None).to_numpy()

                for i in range(binned_concordances0.shape[1]-2):
                    plt.subplot(1,3,1)
                    plt.plot(binned_concordances0[:,0], binned_concordances0[:,i+1], label = "AF <" + str(bins[i,0] * 100) + "%" , linewidth = 2)
                    plt.title("0",fontsize = 10)
                    plt.xlabel("Epoch")
                    plt.ylabel("Concordance")
                    plt.subplot(1,3,2)
                    plt.plot(binned_concordances1[:,0], binned_concordances1[:,i+1], label = "AF <" + str(bins[i,0] * 100) + "%", linewidth = 2)
                    plt.title("1",fontsize = 10)
                    plt.xlabel("Epoch")
                    plt.ylabel("Concordance")
                    plt.subplot(1,3,3)
                    plt.plot(binned_concordances2[:,0], binned_concordances2[:,i+1], label = "AF <" + str(bins[i,0] * 100) + "%", linewidth = 2)
                    plt.title("2",fontsize = 10)
                    plt.xlabel("Epoch")
                    plt.ylabel("Concordance")
                    plt.ylim((-0.05,1.05))
                plt.tight_layout()
                plt.legend(prop={'size': 8})
                plt.savefig(results_directory+"/" + save_name+ ".pdf")

            if not contrastive and False:

                plot_binned_conc("binned_concordances", "test_binned_conc2")
                plot_binned_conc("binned_concordances_train", "binned_concordances_train")
                plot_binned_conc("binned_concordances_valid", "binned_concordances_valid")

            try:
                plot_genotype_hist(np.array(genotypes_output), "{0}/{1}_e{2}".format(results_directory, "output_as_genotypes", epoch))
                plot_genotype_hist(np.array(true_genotypes), "{0}/{1}".format(results_directory, "true_genotypes"))
            except:
                pass

            ################################################################

            ############################### losses ##############################
            plt.figure()
            outfilename = "{0}/losses_from_project.csv".format(results_directory)
            epochs_combined, losses_train_combined = write_metric_per_epoch_to_csv(outfilename, losses_train, epochs)


            plt.plot(epochs_combined, losses_train_combined,
                    label="all data",
                    c="red")

            plt.xlabel("Epoch")
            plt.ylabel("Loss function value")
            plt.legend()
            plt.savefig(results_directory + "/" + "losses_from_project.pdf")
            plt.close()

            ############################### gconc ###############################
            try:
                baseline_genotype_concordance = get_baseline_gc(true_genotypes)
            except:
                baseline_genotype_concordance = None
            plt.figure()
            outfilename = "{0}/genotype_concordances.csv".format(results_directory)
            epochs_combined, genotype_concs_combined = write_metric_per_epoch_to_csv(outfilename, genotype_concs_train, epochs)

            plt.plot(epochs_combined, genotype_concs_combined, label="train", c="orange")
            if baseline_genotype_concordance:
                plt.plot([epochs_combined[0], epochs_combined[-1]], [baseline_genotype_concordance, baseline_genotype_concordance], label="baseline", c="black")

            plt.xlabel("Epoch")
            plt.ylabel("Genotype concordance")

            plt.savefig(results_directory + "/" + "genotype_concordances.pdf")

            plt.close()


    if (arguments['evaluate'] or arguments['animate'] or arguments['plot'])  :# and isChief:
        if not os.path.isfile(encoded_data_file):

            chief_print("------------------------------------------------------------------------")
            chief_print("Error: File {0} not found.".format(encoded_data_file))
            chief_print("------------------------------------------------------------------------")
            exit(1)

        epochs = get_projected_epochs(encoded_data_file)

        if arguments['epoch']:
            epoch = int(arguments['epoch'])
            if epoch in epochs:
                epochs = [epoch]
            else:
                chief_print("------------------------------------------------------------------------")
                chief_print("Error: Epoch {0} not found in {1}.".format(epoch, encoded_data_file))
                chief_print("------------------------------------------------------------------------")
                exit(1)

        if doing_clustering:
            if arguments['animate']:
                chief_print("------------------------------------------------------------------------")
                chief_print("Error: Animate not supported for genetic clustering model.")
                chief_print("------------------------------------------------------------------------")
                exit(1)


            if arguments['plot'] and not superpopulations_file:
                chief_print("------------------------------------------------------------------------")
                chief_print("Error: Plotting of genetic clustering results requires a superpopulations file.")
                chief_print("------------------------------------------------------------------------")
                exit(1)


    if arguments['animate']:

        print("Animating epochs {}".format(epochs))

        scatter_points_per_epoch = []
        colors_per_epoch = []
        markers_per_epoch = []
        edgecolors_per_epoch = []

        ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")

        for epoch in epochs:
            print("########################### epoch {0} ###########################".format(epoch))

            encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

            coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
            name = ""

            if superpopulations_file:
                scatter_points, colors, markers, edgecolors = \
                    plot_coords_by_superpop(coords_by_pop, name, superpopulations_file, plot_legend=False, savefig=False)
                suffix = "_by_superpop"
            else:
                try:
                    scatter_points, colors, markers, edgecolors = plot_coords_by_pop(coords_by_pop, name, savefig=False)
                    suffix = "_by_pop"
                except:
                    scatter_points, colors, markers, edgecolors = plot_coords(encoded_train, name, savefig=False)
                    suffix = ""

            scatter_points_per_epoch.append(scatter_points)
            colors_per_epoch.append(colors)
            markers_per_epoch.append(markers)
            edgecolors_per_epoch.append(edgecolors)

        make_animation(epochs, scatter_points_per_epoch, colors_per_epoch, markers_per_epoch, edgecolors_per_epoch, "{0}/{1}{2}".format(results_directory, "dimred_animation", suffix))

    if arguments['evaluate']:

        print("Evaluating epochs {}".format(epochs))

        # all metrics assumed to have a single value per epoch
        metric_names = arguments['metrics'].split(",")
        metrics = dict()

        for m in metric_names:
            metrics[m] = []

        ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
        pop_list = []

        for pop in ind_pop_list_train[:, 1]:
            try:
                pop_list.append(pop.decode("utf-8"))
            except:
                pass

        for epoch in epochs:
            print("########################### epoch {0} ###########################".format(epoch))

            encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

            coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

            ### count how many f1 scores were doing
            f1_score_order = []
            num_f1_scores = 0
            for m in metric_names:
                if m.startswith("f1_score"):
                    num_f1_scores += 1
                    f1_score_order.append(m)

            f1_scores_by_pop = {}
            f1_scores_by_pop["order"] = f1_score_order

            for pop in coords_by_pop.keys():
                f1_scores_by_pop[pop] = ["-" for i in range(num_f1_scores)]
            f1_scores_by_pop["avg"] = ["-" for i in range(num_f1_scores)]

            for m in metric_names:

                if m == "hull_error":
                    coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
                    n_latent_dim = encoded_train.shape[1]
                    if n_latent_dim == 2:
                        min_points_required = 3
                    else:
                        min_points_required = n_latent_dim + 2
                    hull_error = convex_hull_error(coords_by_pop, plot=False, min_points_required= min_points_required)
                    print("------ hull error : {}".format(hull_error))

                    metrics[m].append(hull_error)

                elif m.startswith("f1_score"):
                    this_f1_score_index = f1_score_order.index(m)

                    k = int(m.split("_")[-1])
                    # num_samples_required = np.ceil(k/2.0) + 1 + (k+1) % 2
                    num_samples_required = 1

                    pops_to_use = get_pops_with_k(num_samples_required, coords_by_pop)

                    if len(pops_to_use) > 0 and "{0}_{1}".format(m, pops_to_use[0]) not in metrics.keys():
                        for pop in pops_to_use:
                            try:
                                pop = pop.decode("utf-8")
                            except:
                                pass
                            metric_name_this_pop = "{0}_{1}".format(m, pop)
                            metrics[metric_name_this_pop] = []


                    f1_score_avg, f1_score_per_pop = f1_score_kNN(encoded_train, pop_list, pops_to_use, k = k)
                    print("------ f1 score with {0}NN :{1}".format(k, f1_score_avg))
                    metrics[m].append(f1_score_avg)
                    assert len(f1_score_per_pop) == len(pops_to_use)
                    f1_scores_by_pop["avg"][this_f1_score_index] =  "{:.4f}".format(f1_score_avg)

                    for p in range(len(pops_to_use)):
                        try:
                            pop = pops_to_use[p].decode("utf-8")
                        except:
                            pop = pops_to_use[p]

                        metric_name_this_pop = "{0}_{1}".format(m, pop)
                        metrics[metric_name_this_pop].append(f1_score_per_pop[p])
                        f1_scores_by_pop[pops_to_use[p]][this_f1_score_index] =  "{:.4f}".format(f1_score_per_pop[p])

                else:
                    print("------------------------------------------------------------------------")
                    print("Error: Metric {0} is not implemented.".format(m))
                    print("------------------------------------------------------------------------")

            write_f1_scores_to_csv(results_directory, "epoch_{0}".format(epoch), superpopulations_file, f1_scores_by_pop, coords_by_pop)

        for m in metric_names:

            plt.plot(epochs, metrics[m], label="train", c="orange")
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.savefig("{0}/{1}.pdf".format(results_directory, m))
            plt.close()

            outfilename = "{0}/{1}.csv".format(results_directory, m)
            with open(outfilename, mode='w') as res_file:
                res_writer = csv.writer(res_file, delimiter=',')
                res_writer.writerow(epochs)
                res_writer.writerow(metrics[m])

    if arguments['plot']:

        print("Plotting epochs {}".format(epochs))

        ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
        pop_list = []

        for pop in ind_pop_list_train[:, 1]:
            try:
                pop_list.append(pop.decode("utf-8"))
            except:
                pass

        for epoch in epochs:
            print("########################### epoch {0} ###########################".format(epoch))

            encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

            coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

            if superpopulations_file:

                coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

                if doing_clustering:
                    plot_clusters_by_superpop(coords_by_pop, "{0}/clusters_e_{1}".format(results_directory, epoch), superpopulations_file, write_legend = epoch == epochs[0])
                else:
                    scatter_points, colors, markers, edgecolors = \
                        plot_coords_by_superpop(coords_by_pop, "{0}/dimred_e_{1}_by_superpop".format(results_directory, epoch), superpopulations_file, plot_legend = epoch == epochs[0])

            else:
                try:
                    plot_coords_by_pop(coords_by_pop, "{0}/dimred_e_{1}_by_pop".format(results_directory, epoch))
                except:
                    plot_coords(encoded_train, "{0}/dimred_e_{1}".format(results_directory, epoch))


    # Clean up temporary tf record files.
    try: shutil.rmtree("{0}/{1}".format(train_directory, "weights_temp"))
    except: pass

    if "SLURM_JOBID" in os.environ:
        try: shutil.rmtree("./Data/temp"+ os.environ["SLURM_JOBID"]+"/")
        except: pass
    else:
        try: shutil.rmtree("./Data/temp/")
        except: pass

    try: os.remove(superpopulations_file+"_temp")
    except: pass
if __name__ == "__main__":
    main()
