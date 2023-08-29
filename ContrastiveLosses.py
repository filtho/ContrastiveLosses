""" Implementations of some contrastive loss functions.

This file contains some helper functions, and classes that can be instantiated and used to compute loss values in contrastive learning.


Typical usage example: 

    import tensorflow as tf
    import ContrastiveLosses as CL
    anchors = tf.cast(tf.transpose(tf.convert_to_tensor([[0,1,1,2,3,4],[1,0,2,1,2,0]])), tf.float32)  
    positives = anchors+ tf.random.uniform(shape= tf.shape(anchors), minval = 0, maxval = 0.1)

    loss_function = CL.centroid(n_pairs = 3, mode = 'distance_weighted_random', distance ="L2")

    loss = loss_function(anchors, positives)


"""

import tensorflow as tf


def gumbel_max(logits, K):
    """
    Draws samples with weightes probabilities without replacement.
    Inspired by (blatantly taken from) https://github.com/tensorflow/tensorflow/issues/9260

    Args:
        logits: tensor of log-probabilites of each considered sample.
        K: Number of samples to return.
    
    Returns:
        A tensor containing the indices of the sampled elements.


    """

    if tf.shape(logits)[0] == 0:
        return tf.convert_to_tensor([[]], dtype=tf.int32)
    else:

        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
        _, indices = tf.nn.top_k(logits + z, K)
        return indices

@tf.function
def lp_distance(x,y,p):
    """ 
    Compute the pairwise L_p distance between x and y. 
    Assumes that x and y are ordered as [samples, embedding_dimension]

    Args: 
        x, y: two d-dimensional points.
        p: The degree of the norm used.

    Returns: 

    """
    return tf.reduce_sum( (x-y)**p , axis = 1)

@tf.function
def distance_matrix(x,y):
    """ 
    Compute the pairwise L_2 distance for all pairs in x and y. Assumes that x and y are ordered as [samples, embedding_dimension]
    
    Args: 
        x, y: Tensors containing d-dimensional points.

    Returns:
        The pairwise L2 distance between pairs in x and y, as a 2 dimensional tensor.
    """

    return  tf.sqrt(tf.reduce_sum((x[:,:,tf.newaxis] - tf.transpose(y[:,:,tf.newaxis]))**2,axis = 1))

@tf.function
def random_negatives(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples randomly from the negative pool for each samples in anchors.
    
    Args: 
        anchors: Tensor of d-dimensional points considered the anchors.
        negative_pool: Tensor of d-dimensional points of potential negatives.
        n_pairs: How many negatives to use.
    
    Returns: A tensor of n dimensional points to use as negatives. 

    """
    n = tf.shape(negative_pool)[0]
    n_pairs = tf.minimum(n_pairs, n - 1)

    indices = tf.random.uniform(shape=[tf.shape(anchors)[0], n_pairs], minval=0, maxval=tf.shape(negative_pool)[0],
                                dtype=tf.dtypes.int32)
    N = tf.gather(params=negative_pool, indices=indices)

    return N
@tf.function
def negatives_by_distance_random(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples from the negative_pool for each sample in anchors.
    Choice is done by drawing them at random, with weights corresponding to the inverse distance.

    Args: 
        anchors: Tensor of d-dimensional points considered the anchors.
        negative_pool: Tensor of d-dimensional points of potential negatives.
        n_pairs: How many negatives to use.
    
    Returns: A tensor of n dimensional points to use as negatives. 
    """

    n = tf.shape(negative_pool)[0]
    n_pairs = tf.minimum(n_pairs, n - 1)

    distances = distance_matrix(anchors, negative_pool)

    I = tf.eye(tf.shape(distances)[0], tf.shape(distances)[1])
    logodds = tf.math.log((distances + I) ** -1 - I)
    inds = gumbel_max(logodds, n_pairs)
    N = tf.gather(negative_pool, inds)

    return N

@tf.function
def negatives_by_distance(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples from the negative_pool for each sample in anchors.
    Choice is done by drawing only the n_pairs closest ones.

    Args: 
        anchors: Tensor of d-dimensional points considered the anchors.
        negative_pool: Tensor of d-dimensional points of potential negatives.
        n_pairs: How many negatives to use.
    
    Returns: A tensor of n dimensional points to use as negatives. 
    """


    n = tf.math.minimum(tf.shape(anchors)[0], tf.shape(negative_pool)[0])
    n_pairs = tf.minimum(n_pairs, n - 1)

    distances = distance_matrix(anchors, negative_pool)
    argsorted_indices = tf.argsort(distances, axis=1, direction="ASCENDING")
    indices = argsorted_indices[:,1:n_pairs+1]
    N = tf.gather(params = negative_pool, indices = indices)

    return N

def generate_negatives(mode, n_pairs):
    """
    Generate negative samples, using different rules on how the choice is done.

    Args:
        mode: Toggle which function to use when choosing negatives. Currently supported are "random", "closest", "distance_weighted_random".
        n_pairs: How many negatives to use.
    
    Returns:
        A function that can be used to generate negatives, given anchors and a pool of negatives.
    """


    if mode == "random":
        generate_negatives_fun = lambda anchors, negative_pool: random_negatives(anchors, negative_pool, n_pairs)

    elif mode == "closest":
        generate_negatives_fun = lambda anchors, negative_pool : negatives_by_distance(anchors, negative_pool, n_pairs)

    elif mode == "distance_weighted_random":
        generate_negatives_fun = lambda anchors, negative_pool : negatives_by_distance_random(anchors, negative_pool, n_pairs)

    else:
        exit(f"Incorrect mode for the choice of negatives in the loss function. Mode \"{mode}\" is not supported. Only \"random\",\"closest\",\"distance_weighted_random\" are implemented")

    return generate_negatives_fun



class ContrastiveLoss():

    def __init__(self): 
        exit("This is a base class for the implementations, You are not supposed to invoke this...")
 

    def __call__(self, anchors, positives):
            
        if tf.distribute.has_strategy():
            #global_anchors, global_positives = self.gather_samples(anchors, positives)

            #num_devices = tf.distribute.get_replica_context().num_replicas_in_sync

            #loss =  ( self.compute_loss(anchors,positives, tf.stop_gradient(global_anchors)) + self.compute_loss(tf.stop_gradient(global_anchors),tf.stop_gradient(global_positives), anchors) / num_devices  ) / 2
            loss = self.compute_loss(anchors,positives,anchors)
            #loss = self.compute_loss(anchors,positives, tf.stop_gradient(global_anchors))
        else:
            loss = self.compute_loss(anchors,positives,anchors)
    
        return loss


    def gather_samples(self, anchors, positives):
        """
        
        """
        global_anchors = tf.distribute.get_replica_context().all_gather(anchors, axis = 0)
        global_positives = tf.distribute.get_replica_context().all_gather(positives, axis = 0)
        return global_anchors, global_positives



    def compute_loss(self, anchors,positives, negative_pool):

        exit("function compute_loss not implemented for the base class.")
        return -1

class Triplet_loss(ContrastiveLoss):
    
    def __init__(self, alpha = 1., mode = 'random', distance ="L2"):
        self.alpha = alpha
        self.mode = mode


        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")

        self.generate_negatives = generate_negatives(mode = mode, n_pairs = 1)
    @tf.function
    def compute_loss(self, anchors, positives, negative_pool):
        
        negatives = tf.squeeze(self.generate_negatives(anchors, negative_pool), axis = 1)

        anchor_pos = self.distance(anchors, positives)
        anchor_neg = self.distance(anchors, negatives)
        
        loss = tf.reduce_sum(tf.math.maximum( anchor_pos - anchor_neg+ self.alpha, 0 ) )

        return loss


class n_pair(ContrastiveLoss):
    
    def __init__(self, n_pairs = 20, alpha = 1., mode = 'distance_weighted_random', distance ="L2"):
        self.mode = mode

        self.alpha = alpha
        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")


        self.generate_negatives = generate_negatives(mode = mode, n_pairs = n_pairs)
    @tf.function
    def compute_loss(self,anchors, positives, negative_pool):
        """
        Implementation of the the N-pair loss from "Improved Deep Metric Learning with Multi-class N-pair Loss Objective"
        by Sohn, 2016
        """
        if tf.shape(anchors)[0] == 0 or tf.shape(negative_pool)[0] == 0 :  # To not have to handle empty batches which may come up in distributed training.
            return 0.
        A = anchors
        P = positives 
        N = self.generate_negatives(anchors,negative_pool)

        N_pairs = tf.shape(N)[1]
        L2_distance_version = True
        if L2_distance_version == True:
            anchor_pos = tf.reduce_sum((tf.tile(A[:,tf.newaxis,:], [1, N_pairs, 1]) - tf.tile(P[:,tf.newaxis,:], [1, N_pairs, 1]))**2, axis=2)
            anchor_neg = tf.reduce_sum((tf.tile(A[:,tf.newaxis,:], [1, N_pairs, 1]) - N)**2, axis=2)
            
            loss = tf.reduce_sum(tf.math.maximum( anchor_pos - anchor_neg+ self.alpha, 0 ) ) / tf.cast(N_pairs,tf.float32)
        else:
            dot_pos = tf.reduce_sum((A * P), axis=1)
            dot_neg = tf.reduce_sum(tf.tile(A[:,tf.newaxis,:], [1, N_pairs, 1]) * N, axis=2)

            m = tf.math.maximum(dot_pos, tf.reduce_max(dot_neg,axis = -1))

            pos = tf.math.exp(dot_pos -m)
            neg = tf.reduce_sum(tf.math.exp(dot_neg-m[:,tf.newaxis]), axis=1)

            loss = -tf.reduce_sum(tf.math.log(pos / (neg + pos)))

        return loss

class centroid(ContrastiveLoss):
    
    def __init__(self, n_pairs = 20, mode = 'distance_weighted_random', distance ="L2"):
        self.mode = mode


        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")


        self.generate_negatives = generate_negatives(mode = mode, n_pairs = n_pairs)
    @tf.function
    def compute_loss(self,anchors, positives, negative_pool):
        """
        This loss should have only 2D output, and no L2 normalization on the output.
        Input shapes in the single gpu case here is [batch size, dimension of embedding (most cases = 2)]

        Want to have "regular" n-pair loss, but instead of computing vectors from the origin, compute
        them with respect to some other point. We talked first about doing this with respect to the
        mean coordinate of all the samples considered, but one thought I had was that, for large N, this
        would in mean result in just using __approximately__ the origin.
        Carl had an idea to use different centroids for each considered negative for each sample.
        The centroid would then sort of be "(A+P+2*N)/4", weighting N twice since most likely A and P will lie close.
        
        The loss function is sum_samples(log (1 + sum_negatives( exp( (A-C)'*(A-N)- (A-C)'* (A-P)  ))) )
       
        """

        n =  tf.shape(anchors)[0]
        if tf.shape(anchors)[0] == 0 or tf.shape(negative_pool)[0] == 0 :  # To not have to handle empty batches which may come up in distributed training.
            return 0.

        A = anchors
        P = positives 
        N = self.generate_negatives(anchors,negative_pool)

        A_full = tf.tile(A[:, tf.newaxis, :], [1, tf.shape(N)[1], 1])
        P_full = tf.tile(P[:, tf.newaxis, :], [1, tf.shape(N)[1], 1])
        
        C = (A_full + P_full + 2 * N) / 4  

        AC = A_full - C
        NC = N - C
        PC = P_full - C

        max_vec = tf.tile(tf.reduce_max(tf.stack(
            [tf.reduce_sum(AC ** 2, axis=-1), tf.reduce_sum(NC ** 2, axis=-1), tf.reduce_sum(PC ** 2, axis=-1)], axis=-1),
            axis=-1)[:, :, tf.newaxis], [1, 1, tf.shape(anchors)[1]])
        
        eps = 1e-12

        num = tf.reduce_sum(AC * NC / (max_vec+eps), axis=2)
        denom = tf.reduce_sum(AC *PC / (max_vec+eps), axis=2)

        m = tf.reduce_max(tf.stack([num, denom], axis=-1), axis=-1)

        loss = tf.reduce_sum(tf.math.log(
            1 + tf.reduce_sum(-tf.math.exp(-2.) + tf.math.exp(num - m) / (tf.math.exp(denom - m) + eps), axis=1)))#  *  1 / tf.cast(n, tf.float32)


        return loss


class debiased_contrastive_loss(ContrastiveLoss):
    
    def __init__(self, n_pairs = 20, mode = 'distance_weighted_random', distance ="L2"):
        
        self.mode = mode
        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")


        self.generate_negatives = generate_negatives(mode = mode, n_pairs = n_pairs)
    @tf.function
    def compute_loss(self,anchors, positives, negative_pool):
        """
        This implements the "Debiased contrastive loss", as presented in "Debiased Contrastive Learning", Chuang et.al. 2020.

        This should essentially assume that the output is L2 normalized, that is, the output domain is on the d-dimensional hypersphere.
        """
        if tf.shape(anchors)[0] == 0 or tf.shape(negative_pool)[0] == 0 :  # To not have to handle empty batches which may come up in distributed training.
            return 0.
      
        A = anchors
        P = positives 
        N = self.generate_negatives(anchors,negative_pool)

        tf.print(tf.shape(N))
        N_pairs = tf.shape(N)[1] 
        dot_pos = tf.reduce_sum((A * P), axis=1) 
        dot_neg =tf.reduce_sum(tf.tile(A[:,tf.newaxis,:], [1, N_pairs, 1]) * N, axis=2)

        m = tf.math.reduce_max([tf.reduce_max(dot_neg),tf.reduce_max(dot_pos)])

        neg = tf.reduce_sum(tf.math.exp(dot_neg-m), axis=1)
        pos = tf.math.exp(dot_pos-m)

        tau_plus = 0.01
        t = 1.
        eps = 1e-12
        N_pairs2 = tf.cast(N_pairs, tf.float32)


        Ng = tf.math.maximum( (-N_pairs2 * tau_plus * pos +neg) / (1-tau_plus), N_pairs2* tf.math.exp(-1/t-m))
        debiased_loss = -tf.reduce_sum(tf.math.log(pos / (pos+ Ng+ eps )))

        return debiased_loss




