import glob
import math
import os.path
import tensorflow as tf

#Functions in this file is mostly based on https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
# and tensorflow docs


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'len': tf.io.FixedLenFeature([], tf.int64),
        'snp': tf.io.FixedLenFeature([], tf.string),
        'identity': tf.io.FixedLenFeature([], tf.string),
        'pop': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)

    snp = content['snp']
    identity = content['identity']
    pop = content['pop']
    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(snp, out_type=tf.float32)
    identity = tf.io.parse_tensor(identity, out_type=tf.string)
    pop = tf.io.parse_tensor(pop, out_type=tf.string)

    return feature, [identity, pop]


def parse_single_example(data, identity, pop):
    # define the dictionary -- the structure -- of our single example
    data = {
        'len': _int64_feature(data.shape[0]),
        'snp': _bytes_feature(serialize_array(data)),
        'identity': _bytes_feature(serialize_array(identity)),
        'pop': _bytes_feature(serialize_array(pop)),

    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_genotypes_to_tfr_long_one_sample(data, dataset, filename="data_tfrecord", n_workers=1, out_dir="./",
                                mode="training"):
    if mode == "training" or mode == "project":
        n_samples = data.n_train_samples
    elif mode == "validation":
        n_samples = data.n_valid_samples
    else:
        print("Choose a correct mode (training or validation)")
        exit(3)

    berra_scratch_dir = "/scratch2/temp/"
    if not os.path.isdir("Data/temp/"):
        os.mkdir("Data/temp/")




    if "SLURM_JOBID" in os.environ:
        if not os.path.isdir(berra_scratch_dir):
            os.mkdir(berra_scratch_dir)

        if not os.path.isdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/" ):
            os.mkdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/")



        if os.path.isdir(berra_scratch_dir+os.environ["SLURM_JOBID"] +"/" +mode):
            return
    else:
        if not os.path.isdir("Data/temp/"):
            os.mkdir("Data/temp/")

        if os.path.isdir("Data/temp/"+mode):
            return

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
   
    current_shard_nr = 0
    
    for genotypes, pops, _ in dataset:
        
        #tf.print(tf.shape(pops))
        #tf.print(tf.shape(genotypes))

        # Tensorflow version 2.12 seemed to have issues with throwing spurious errors. This check magically makes it work.
        assert(tf.shape(genotypes)[0] == tf.shape(pops)[0])

        for ii, geno in enumerate(genotypes):
            #enos = geno
            #pops = pop[ii,:]
            num_zeros = 7
            #tf.print(ii)
            #assert ii < tf.shape(pops)[0]

            current_shard_nr += 1

            num_zeros -= len(str(current_shard_nr))
            identity = "0" * num_zeros + str(current_shard_nr)
            current_shard_name = "{}/{}_{}{}{}.tfrecords".format(out_dir, identity, n_samples, mode, filename)
            writer = tf.io.TFRecordWriter(current_shard_name)
            out = parse_single_example(data=geno, identity=pops[ii,0], pop=pops[ii,1])
            writer.write(out.SerializeToString())


def write_genotypes_to_tfr_long_one_sample_mult_dir(data, dataset, filename="data_tfrecord", n_workers=1, out_dir="./",
                                mode="training"):
    if mode == "training" or mode == "project":
        n_samples = data.n_train_samples
    elif mode == "validation":
        n_samples = data.n_valid_samples
    else:
        print("Choose a correct mode (training or validation)")
        exit(3)

    berra_scratch_dir = "/scratch2/temp/"
    if not os.path.isdir("Data/temp/"):
        os.mkdir("Data/temp/")

    if "SLURM_JOBID" in os.environ:
        if not os.path.isdir(berra_scratch_dir):
            os.mkdir(berra_scratch_dir)
        
        if not os.path.isdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/" ):
            os.mkdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/")
        if os.path.isdir(berra_scratch_dir+os.environ["SLURM_JOBID"] +"/" +mode):
            return
    else:
        if not os.path.isdir("Data/temp/"):
            os.mkdir("Data/temp/")

        if os.path.isdir("Data/temp/"+mode):
            return

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    samples_per_dir = 1000
    current_shard_nr = 0

    for genotypes, pops, _ in dataset:

        # Tensorflow version 2.12 seemed to have issues with throwing spurious errors. This check magically makes it work.
        assert(tf.shape(genotypes)[0] == tf.shape(pops)[0])

        for ii, geno in enumerate(genotypes):

            if not os.path.isdir(f"{out_dir}{current_shard_nr//samples_per_dir}"):
                os.mkdir(f"{out_dir}{current_shard_nr//samples_per_dir}")

            num_zeros = 7
            num_zeros -= len(str(current_shard_nr))
            identity = "0" * num_zeros + str(current_shard_nr)
            current_shard_name = f"{out_dir}/{current_shard_nr//samples_per_dir}/{identity}_{n_samples}{mode}{filename}.tfrecords"
            writer = tf.io.TFRecordWriter(current_shard_name)
            out = parse_single_example(data=geno, identity=pops[ii,0], pop=pops[ii,1])
            writer.write(out.SerializeToString())
            
            current_shard_nr += 1

def write_genotypes_to_tfr_long(data, dataset, filename="data_tfrecord", n_markers=0, n_workers=1, out_dir="./",
                                mode="training"):
    if mode == "training" or mode == "project":
        n_samples = data.n_train_samples
    elif mode == "validation":
        n_samples = data.n_valid_samples
    else:
        print("Choose a correct mode (training or validation)")
        exit(3)

    berra_scratch_dir = "/scratch2/temp/"
    if not os.path.isdir("Data/temp/"):
        os.mkdir("Data/temp/")

  


    if "SLURM_JOBID" in os.environ:
        if not os.path.isdir(berra_scratch_dir):
            os.mkdir(berra_scratch_dir)
        
        if not os.path.isdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/" ):
            os.mkdir(berra_scratch_dir+ os.environ["SLURM_JOBID"] +"/")

        if os.path.isdir(berra_scratch_dir+os.environ["SLURM_JOBID"] +"/" +mode):
            return
    else:
        if not os.path.isdir("Data/temp/"):
            os.mkdir("Data/temp/")

        if os.path.isdir("Data/temp/"+mode):
            return

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    tf.print(os.path.isdir("Data/temp/"+mode))
    batch_size = data.batch_size// n_workers
    tf.print(n_workers)

    # Choose number of shards so that each file is at least 10 MB large, but at most 10 files per worker, saving using 4 bit numbers.
    shards_per_worker = min(10, math.ceil(n_samples * n_markers * 1e-6 * 4 / (10 * n_workers)))

    resulting_file_size = n_samples * n_markers * 1e-6 * 4 / (shards_per_worker * n_workers)

    # correcting for having each file having integer multiple of batch_size
    samples_per_file = int(resulting_file_size / 4 * 1e6 / n_markers)
    samples_per_file_corrected = batch_size * (samples_per_file // batch_size) + batch_size * (
                (samples_per_file // batch_size) == 0)



    total_number_of_files = math.ceil(n_samples / samples_per_file_corrected)
    splits = total_number_of_files  # + total_number_of_files%n_workers # Just so that each worker has same number of files (some may be empty).

    batches_per_file = samples_per_file_corrected // batch_size

    file_count = 0
    total_index = 0

    count = 0
    current_shard_nr = 0
    for genotypes, pop, last_batch in dataset:

        if count == 0:
            genos = genotypes
            pops = pop
        else:
            genos = tf.concat([genos, genotypes], axis=0)
            pops = tf.concat([pops, pop], axis=0)
        count += 1

        if count == batches_per_file or last_batch:
            count = 0
            num_zeros = 4

            current_shard_nr += 1

            num_zeros -= len(str(current_shard_nr))
            identity = "0" * num_zeros + str(current_shard_nr)

            if "SLURM_LOCALID" in os.environ:  # This is made as to not have several processes trying to create the same file, which results in a corrupted file. Could most likely speed this up! Let them each write
                if int(os.environ["SLURM_LOCALID"]) == 0:

                    current_shard_name = "{}/{}_{}{}{}.tfrecords".format(out_dir, identity, splits, mode, filename)
                else:
                    current_shard_name = "{}/{}_{}{}{}{}.tfrecords".format(out_dir, identity, splits, mode, filename, "TEMP")
            else:
                current_shard_name = "{}/{}_{}{}{}.tfrecords".format(out_dir, identity, splits, mode, filename)

            # if os.path.exists(current_shard_name):
            # return
            writer = tf.io.TFRecordWriter(current_shard_name)

            for index in range(batch_size * batches_per_file):
                # while current_shard_count < (batches_per_file * data.batch_size):  # as long as our shard is not full
                # get the index of the file that we want to parse now
                # index = current_shard_count #+ i * (batches_per_file * data.batch_size)

                current_genotype = tf.cast(genos[index, :], tf.float32)

                # create the required Example representation
                out = parse_single_example(data=current_genotype, identity=pops[index, 0], pop=pops[index, 1])

                writer.write(out.SerializeToString())
                total_index += 1
                              
                if total_index == n_samples:  # when we have consumed the whole data, preempt generation
                    break

            file_count += 1

            writer.close()
    tf.print("wrote " + str(file_count))
    if "SLURM_JOBID" in os.environ: 

        tf.print(int(os.environ["SLURM_PROCID"]) , current_shard_name)
    

    return (batches_per_file * batch_size)


def get_dataset_large1(tfr_dir="./", pattern="*human_origins_tfr.tfrecords"):
    files = glob.glob(tfr_dir + pattern, recursive=False)

    # create the dataset
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=10)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


def get_dataset_large(tfr_dir="./", pattern="*human_origins_tfr.tfrecords"):  # ,device_id = 0):

    files = glob.glob(tfr_dir + pattern, recursive=False)
    files.sort()

    # d = tf.data.Dataset.list_files(tfr_dir + pattern)
    d = tf.data.Dataset.from_tensor_slices(files)
    if "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
        num_workers = int(os.environ["SLURM_NTASKS"])
    else:
        worker_index = 0
        num_workers = 1 # Asssume only one worker per machine.

    shuffle_buffer_size = len(files)+1 
    d = d.shuffle(shuffle_buffer_size)# This shuffles the files before sharding. This makes is so that the files are not stuck on a specific process, but are moved around.

    d = d.shard(num_workers, worker_index)

    num_readers = num_workers

    # Now, look into how much slower if we have one sample per file.

    # Dont shuffle the files, shuffle the samples, as for now, this may have an adverse effect in the effective space of recombined individuals
    # Ideally, we would also want the shuffle the files,
    # In addition,  this may be very bad for contrastive learning, each sample should be able to see all other samples, now there will be some sort of partitioning. This is bad.

    # Pretty bad first solution to this, would be to shuffle the files, and re-refining dataset each epoch. Doing this has not really been seen to decrease performance, tested when 
    # looking into "coordinate-based recombination choices".    

    d = d.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE,
                     cycle_length=num_readers, block_length=1)

    d = d.map(parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)


    return d   #, d2


def get_dataset_large_mult_dir(tfr_dir="./", pattern="*human_origins_tfr.tfrecords"):
    
    dirs = os.listdir(tfr_dir)

    files_full =[]
    for directory in dirs:

        files = glob.glob(tfr_dir + directory+ "/" + "*.tfrecords" , recursive=False)
        files_full.extend(files)

    files_full.sort()

    d = tf.data.Dataset.from_tensor_slices(files_full)
    if "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
        num_workers = int(os.environ["SLURM_NTASKS"])
    else:
        worker_index = 0
        num_workers = 1 # Asssume only one worker per machine.

    shuffle_buffer_size = len(files_full)+1
    d = d.shuffle(shuffle_buffer_size)# This shuffles the files before sharding. This makes is so that the files are not stuck on a specific process, but are moved around.

    d = d.shard(num_workers, worker_index)
    num_readers = num_workers

    d = d.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE,
                     cycle_length=num_readers, block_length=1)

    d = d.map(parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)


    return d  
