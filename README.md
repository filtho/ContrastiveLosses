# ContrastiveLosses
Implementations and examples of use cases of loss functions used in contrastive representation learning. The implementations of the loss functions are found in contrastive_losses.py. 

Implemented in TensorFlow, orginally for use in genetic data, here shown for other applications as well.
### Computational Environment

#### Singularity:
The required packages are all included in the Apptainer definition file `image.def`.

Build the Apptainer image:

`$ sudo apptainer build  image.sif image.def`

Run the image, the --nv flag exposes the NVIDIA GPU to the container: 

`$ apptainer run --nv image.sif`

### Virtual environment specification
As an alternative, the file requirements.txt contains the needed packages. Creating a python venv, and executing 
`pip install -r requirements.txt`
will install the packages needed to run the code.



# Example on Genetic data:


## Getting the data:
The data used in this example is described in [this paper](https://pubmed.ncbi.nlm.nih.gov/28445722/), and the data can be obtained by running 

`$ wget ftp://ftp.nhgri.nih.gov/pub/outgoing/dog_genome/SNP/2017-parker-data/*`

It contains SNP data oon ~1300 dogs from 23 clades, woth ~150k variants.

Place these files in the Data folder, and then run the above commands. The accompanying file dog_superpopulations corresponds to this specific dataset.

## Command line interface


The program `run_gcae.py` is called for examples of contrastive learning on genetic data.

To run, the user need to state whether we want to train a model anew or to project already saved model states, among other parameters. This project is a continuation of [GenoCAE](https://github.com/kausmees/GenoCAE), see that page for a more detailed usage guide. This project shares essentially 
the same API.


For example, to train a model on the dog dataset, execute the following:

`$ python run_gcae.py train --trainedmodeldir=./test  --datadir=Data/dog --model_id=CM_M1_2D --data=All_Pure_150k --train_opts_id=ex3_CL --data_opts_id=d_0_4_dog_cont --save_interval=5 --epochs=100`

To plot results for saved model states in a directory and evaluate the KNN-classification accuracy, run 

`$python run_gcae.py project --trainedmodeldir=./test  --datadir=Data/dog --model_id=CM_M1_2D --data=All_Pure_150k --train_opts_id=ex3_CL --data_opts_id=d_0_4_dog_cont --superpops=Data/dog/dog_superpopulations`


![Results on Dog dataset](gcae/animated.gif)


## Some notes:

Depending on the hardware setup, some minor changes may need to be made for the code to run. One issue could be the GPU running out of memory. Reducing the batch size in the train_opts file could be one fix. For my current setup, I had to explicitly allocate more memory than tensorflow automatically did. This can be done by adding the line 
`tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])`
a VRAM of 3.5GB should be enough to run the example with the current settings. On my machine, one training epoch takes ~25 seconds.


# Examples for image data
## Command line interface
The program `run_CL.py` is called for all implemented examples, at the time of writing MNIST and Cifar10.

To run, the user need to state whether we want to train a model anew or to plot already saved model states, which dataset to run, and where to save the logs.
For example, to train a model on the MNIST dataset, execute the following:

`$ python3 -u run_CL.py train --data=mnist --dir=./test_mnist`

There is also an optional argument to restart training from a previously saved state, by appending the `--load_path=/path_to_saved_model` argument.

To plot results for saved model states in a directory and evaluate the KNN-classification accuracy, run 

`$ python3 -u run_CL.py plot --data=mnist --dir=./test_mnist`
Currently, this also runs PCA and t-SNE to compare with. Note that t-SNE may be relatively slow to run.



### Examples
#### MNIST
Within the singularity container running the example of contrastive learning on the MNIST dataset as described above for approximately 400 epochs, generates the following output: ![Results on Mnist](example_figures/mnist_example.jpg)


#### Fashion-MNIST

A similar dataset in size, but with clothing items instead is the Fashion-MNIST dataset. 

`$ python3 -u run_CL.py train --data=fashion_mnist --dir=./test_fashion_mnist`
The code yields the following results:


![Results on Fashion-MNIST](example_figures/fashion_mnist_example.png)


