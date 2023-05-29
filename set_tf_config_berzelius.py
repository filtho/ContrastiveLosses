import os
import json
import csv



def set_tf_config():
    """
        This functions sets the necessary values in the TF_CONFIG environment variable. It contains information on the
        cluster architectures, i.e, which workers are allocated for the job, and the worker for the current task. This has
        been specifically been developed for using the SLURM task manager, and kind of hardcoded for the ouput given using
        the Berzelius supercomputer at NSC. It may be the case that this funciton does not work on other clusters without
        some changes.

        Here, the outputted string s from the call  to os.environ["SLURM_JOB_NODELIST"], contains all the allocated
        workers for the job.

            Examples:
                s = "Node021"
                s = "Node[036-039]"
                s = "Node[009-012,047]"

            We need to translate this format into a separated list of strings representing the nodes to be able to
            describe the cluster in a way that tf.distribute.MultiWorkerMirroredStrategy can interpret the cluster. That
            is we want:
                s = "Node021"             -> ["Node021"]
                s =" Node[036-039]"       -> ["Node036", "Node037", "Node038", "Node039"]
                s = "Node[009-012,047]"   -> ["Node009","Node010","Node011","Node012", "Node047"]

            This is what is done below.
            An example for the case s = Node[009-012,047] is followed within the comments.

            UPDATE: Some experiments indicate that I would want to have one process per gpu, instead of one process per node.

            I can't imagine that I would need to make drastic changes, but this script needs to be updated according to this.

            Probably the easiest case to look at is when we have the same number of GPUS on each node, but for a more general
            implementation, we should be able to have for example 8 on one, and just 1 on another.
            This can be handled as is, but want it to work for the case with 1 process per GPU.

            Start by at least assuming that we have the same number of gpus per node, I would have to make something special in the sbatch call if I would
            want to have say 7+5 gpus. Should possibly work in future.
    """
    s = os.environ["SLURM_JOB_NODELIST"]  #example:  s = "Node[009-012,047]"
    #print(os.environ["CUDA_VISIBLE_DEVICES"] )
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" # os.environ["SLURM_LOCALID"]
    #print("GPUS : ", os.environ["CUDA_VISIBLE_DEVICES"] )


    if s.find("[") == -1:  # The case with only one node, has no brackets. Finds the node number.
        s4 = [str(s[s.find("e") + 1:])]

    else:
        s2 = s[s.find("[") + 1:  s.find("]")]  # s2 = "009-012,047"

        s3 = s2.split(",")  # s3 = ["009-012","047"]
        s4 = []
        for i in s3:
            j = i.find("-")
            if j != -1:
                s5 = i.split("-")
                a = int(s5[0])
                while a < int(s5[1]) + 1:
                    s4.append(str(a))
                    a += 1
            else:
                s4.append(i)  # s3 = ["009","010","011","012","047"]
        #print(s4)

    # The node numbering is done using three digits, padded with zeros if necessary.
    number_of_zeros = [3 - len(i) for i in s4]
    clust = ["node" + "0" * i[0] + i[1] for i in zip(number_of_zeros, s4)]  # Clust =  ["Node009","Node010","Node011","Node012", "Node047"]

    # All of the above should hold most likely

    # Now, I want to know how many tasks we have. Could possibly want to use the env variable SLURM_TASKS_PER_NODE

    # This may be good as well SLURM_LOCALID

    # I assume that I would need to open different ports for different processes


    port ="888"  # Choose a port number to use, use 888 as base, the processes will then have ports 8880, 8881, 8882, etc

    
    port ="8"+ os.environ["SLURM_JOBID"][-3:]  # New version, the port number will now be 8zzy, where zz is the last two digits in the jobID, and y being the local task id

    # In order to communicate, the nodes need to be supplied with port numbers (This is something that I do not 
    # really understand). 
    
    #clust_with_ports = [s + ":"+port for s in
     #                   clust]  # = ["Node009:8888","Node010:8888","Node011:8888","Node012:8888", "Node047:8888"]

    # This outputs the node used for the specific task, where most likely we want to have 1 node corresponding to 1 
    # task. Use this to check if it is the first worker. The first worker is usually appointed some extra tasks in 
    # addition to training. This can for example be printing stuff etc, just using print() will print using all 
    # tasks, and we will just get extra print statements.
    
    #num_workers = len(clust_with_ports)
    
    num_tasks =  int(os.environ["SLURM_NTASKS"]) // len(clust)  # Here we assume that the  number of GPUS is the same across all nodes.

    ## This should really be the only thing that needs to be changed in order to handle different number of gpus on different nodes
    # I can't find a smart way of finding the number of gpus alocated per node, if this number is different for different nodes.
    #  SLURM_TASKS_PER_NODE=2(x3),1, translate this into a list or array of [2,2,2,1], then loop over this.

    string = os.environ["SLURM_TASKS_PER_NODE"]
    string2 = string.split(",")

    num_gpus_per_node = []
    for i in string2:
        ind = int((i.find("x")))
        if not ind == -1:
            for j in range(int(i[ind+1])):
                num_gpus_per_node.append(int(i[0]))
        else:
            num_gpus_per_node.append(int(i[0]))


    clust_with_ports = []

    for i in range(len(clust)):
        for j in range(num_gpus_per_node[i]):
            #print(i)
            clust_with_ports.append(clust[i]+":"+port+str(j)) 
            #print(clust[i])

    num_workers = len(clust_with_ports)

    if int(os.environ["SLURM_PROCID"])  ==0 :

        print(clust_with_ports)

    clust_with_ports = []

    for i in range(len(clust)):
        for j in range(num_gpus_per_node[i]):
            #print(i)
            clust_with_ports.append(clust[i]+":"+port+str(j)) 
            #print(clust[i])

    num_workers = len(clust_with_ports)


    if int(os.environ["SLURM_PROCID"])  ==0 :

        print(clust_with_ports)

    
    t = os.environ["SLURMD_NODENAME"]
    # Find at which index the current Node is, if it is the first node in the job, this is appointed chief status. 
    # This is also used as an output from this function 
    ind = clust.index(t)* num_tasks + int(os.environ["SLURM_LOCALID"])
    #print("ind: ", ind)
    if ind == 0 and int(os.environ["SLURM_PROCID"])==0:
        role = "worker"
        chief = t
    else:
        role = "worker"
        ind = ind
        chief = 0

    """
    If we explicitly appoint a worker as the chief, it seems to not take part in training. This can be done in this manner:

        cfg = {'cluster': {'chief': [clust_with_ports[0]], 'worker' : clust_with_ports[1:] },
                #'cluster': {'worker': clust},
                'task': {'type': role,'index': ind,},
                'rpc_layer': 'grpc',}

    Here I say that the first node is the chief, and the rest are workers, i.e., first node does no computational work. This is most likely nto what I want.
    I want all to be working, but I want worker with index 0 to in addition be responsible for printing etc.
    """

    cfg = {
        'cluster': {'worker': clust_with_ports},
        'task': {'type': role, 'index': ind, },
        'rpc_layer': 'grpc' }

    # These addresses with the  "grpc://" prefixes are needed when doing profiling (I think)- profiling multiple 
    # workers seems hard. 
    
    #addresses = [",".join(["grpc://" + c + ":"+port for c in clust])]
    addresses = [",".join(["grpc://" +str(ind)+ c for c in clust_with_ports])]
    addresses = [",".join(["grpc://" + c for c in clust_with_ports])]

    #print(addresses)
    # Now we have the full tf config variable, write it to the os.environ to set it as a environment variable.
    os.environ['TF_CONFIG'] = json.dumps(cfg)

    return addresses, chief, num_workers
