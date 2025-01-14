


# ContrastiveLosses
Implementations and examples of use cases of loss functions used in contrastive representation learning. The loss functions are found in contrastive_losses.py. 

Implemented in TensorFlow, orginally for use in genetic data, here shown for other applications as well.
### Computational Environment
This has been developed and tested only for Linux, not guaranteed to work for Windows.
#### Singularity:
The required packages are all included in the Apptainer definition file `image.def`.

Build the Apptainer image:

`$ sudo apptainer build  image.sif image.def`

Run the image, the --nv flag exposes the NVIDIA GPU to the container: 

`$ apptainer run --nv image.sif`

Python3.10 is installed in this container, so if you use this, you might need to explicitly run with Python version 3.10, so in the example below on genetic data, run 

`$ python3.10 run_gcae.py train ... `

### Virtual environment specification
As an alternative, the file requirements.txt contains the needed packages. Creating a python venv, and executing 
`pip install -r requirements.txt`
will install the packages needed to run the code.



# Example on Genetic data:

This section gives an example use on a data set consisting of dog genotypes, with results presented as a [poster](https://filtho.github.io/poster_pag_30.pdf) at PAG30.

## Getting the data:
Generally, the code supports data in PLINK format. 
You can use your own PLINK data, but the data used in the below example is described in [this paper](https://pubmed.ncbi.nlm.nih.gov/28445722/), and the data can be obtained by running 

`$ wget ftp://ftp.nhgri.nih.gov/pub/outgoing/dog_genome/SNP/2017-parker-data/*`

It contains SNP data on ~1300 dogs from 23 clades, with ~150k variants.

Place these files in the Data folder, and then run the below commands for training and projection of the samples. The accompanying file dog_superpopulations corresponds to this specific dataset.

## Command line interface


The program `run_gcae.py` is called for examples of contrastive learning on genetic data.

To run, the user need to state whether we want to train a model anew or to project already saved model states, among other parameters. This project is a continuation of [GenoCAE](https://github.com/kausmees/GenoCAE), see that page for a more detailed usage guide. This project shares essentially the same API.


For example, to train a model on the dog dataset, run the following. This is an example using all samples in training, with the first 10k SNPs, and a 2D embedding model:

`$ python3 run_gcae.py train --trainedmodeldir=./test  --datadir=Data/dog --model_id=CM_2D_test --data=All_Pure_150k --train_opts_id=ex3_CL --data_opts_id=d_0_4_dog_cont --save_interval=5 --epochs=100`

To plot results for saved model states in a directory, run 

`$ python3 run_gcae.py project --trainedmodeldir=./test  --datadir=Data/dog --model_id=CM_2D_test --data=All_Pure_150k --train_opts_id=ex3_CL --data_opts_id=d_0_4_dog_cont --superpops=Data/dog/dog_superpopulations`


![Results on Dog dataset](gcae/animated.gif)

## Settings in the manuscript
The above example model has a 2-dimensional output. The model `Contrastive3D.json` is the one used in the [preprint](https://www.biorxiv.org/content/10.1101/2024.09.30.615901v1.full.pdf) , and has a normalized 3-dimensional output.

The dog and Human Origins in the manuscript have used the data opts files `d_0_4_dog_filtered.json` and `d_0_4_human.json`, and the train_opts files  `ex3_CL_dog3D.json` and `ex3_CL_human3D.json`, respectively.

The data used is referred to their respective sources, 
The evaluation metrics used to evaluate the embeddings and the plots found in the manuscript are found in `evaluation_scripts/embedding_evaluations.py`. The t-SNE and UMAP embeddings are created with calls from the file `evaluation_scripts/umap_and_tsne.py`

## Some notes:

Depending on the hardware setup, some minor changes may need to be made for the code to run. One issue could be the GPU running out of memory. Reducing the batch size in the train_opts file could be one fix, another would be to use less variants, toggled in the data_opts file. For my current hardware setup, I had to explicitly allocate more memory than tensorflow automatically did. This can be done by adding the line 
`tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])`
a VRAM of 3.5GB should be enough to run the example with the current settings. On my machine, one training epoch takes ~25 seconds for the full dataset, which has been run for the above example.




# PAG 31


## To see other related visualizations, check out [this website.](https://filtho.github.io)

This repo contains the code that has produced the embeddings using contrastive learning, as shown at PAG31, [poster number 676](https://filtho.github.io/poster.pdf) and at ICQG7, [poster number 92](https://filtho.github.io/poster_icqg7.pdf) (day 2).
The poster presented at PAG 31 contained some smaller errors. Most notably in Figure 3 shows PCA with 10 dimensions, instead of 2.
![errata image figure 3](example_figures/test_errata.png)
