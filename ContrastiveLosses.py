import tensorflow as tf

def gumbel_max(logits, K):
    # Inspired by (blatantly taken from) https://github.com/tensorflow/tensorflow/issues/9260
    if tf.shape(logits)[0] == 0:
        return tf.convert_to_tensor([[]], dtype=tf.int32)
    else:

        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
        _, indices = tf.nn.top_k(logits + z, K)
        return indices


def lp_distance(x,y,p):
    """ 
    Compute the pairwise L_p distance between x and y. Assumes that x and y are ordered as [samples, embedding_dimension]
    """
    return tf.reduce_sum( (x-y)**p , axis = 1)

def distance_matrix(x,y):
    """ 
    Compute the pairwise L_2 distance for all pairs in x and y. Assumes that x and y are ordered as [samples, embedding_dimension]
    """

    return  tf.sqrt(tf.reduce_sum((x[:,:,tf.newaxis] - tf.transpose(y[:,:,tf.newaxis]))**2,axis = 1))


def random_negatives(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples randomly from the negative pool for each samples in anchors.
    """
    n = tf.math.minimum(tf.shape(anchors)[0], tf.shape(negative_pool)[0])
    n_pairs = tf.minimum(n_pairs, n - 1)

    indices = tf.random.uniform(shape=[tf.shape(anchors)[0], n_pairs], minval=0, maxval=tf.shape(negative_pool)[0],
                                dtype=tf.dtypes.int32)
    N = tf.gather(params=negative_pool, indices=indices)
    
    return N

def negatives_by_distance_random(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples from the negative_pool for each sample in anchors. Choice is done by drawing them at random, with weights corresponding to the inverse distance.
    """

    n = tf.math.minimum(tf.shape(anchors)[0], tf.shape(negative_pool)[0])
    n_pairs = tf.minimum(n_pairs, n - 1)

    distances = distance_matrix(anchors, negative_pool)

    I = tf.eye(tf.shape(distances)[0], tf.shape(distances)[1])
    logodds = tf.math.log((distances + I) ** -1 - I)
    inds = gumbel_max(logodds, n_pairs)
    N = tf.gather(negative_pool, inds)

    return N


def negatives_by_distance(anchors, negative_pool, n_pairs):
    """
    Draws n_pairs negative samples from the negative_pool for each sample in anchors.
    Choice is done by drawing only the n_pairs closest ones.
    """


    n = tf.math.minimum(tf.shape(anchors)[0], tf.shape(negative_pool)[0])
    n_pairs = tf.minimum(n_pairs, n - 1)

    distances = distance_matrix(anchors, tf.concat([anchors,anchors], axis = 0))
    argsorted_indices = tf.argsort(distances, axis=1, direction="ASCENDING")
    indices = argsorted_indices[:,1:n_pairs+1]
    N = tf.gather(params = negative_pool, indices = indices)

    return N


def generate_negatives(mode, n_pairs):



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
        exit("This is a base class for the implementations, You are not supposed to do this...")

        return 

        
    def __call__(self, anchors, positives): 
            
        if tf.distribute.has_strategy():
            global_anchors, global_positives = self.gather_samples(anchors, positives)

            loss =  ( self.compute_loss(anchors,positives, tf.stop_gradient(global_anchors)) + self.compute_loss(tf.stop_gradient(global_anchors),tf.stop_gradient(global_positives), anchors)  ) / 2

        else:
            loss = self.compute_loss(anchors,positives,anchors)    
    
        return loss


    def gather_samples(self, anchors, positives):
        strategy=tf.distribute.get_strategy()
        global_anchors = tf.distribute.get_replica_context().all_gather(anchors, axis = 0)
        global_positives = tf.distribute.get_replica_context().all_gather(positives, axis = 0)
        return global_anchors, global_positives

    def loss(self,A,P,N):

        return 1

    def compute_loss(self, anchors,positives, negative_pool):

        exit("function compute_loss not implemented for the base class.")
        return -1

class Triplet_loss(ContrastiveLoss):
    
    def __init__(self, alpha = 1, mode = 'random', distance ="L2"):
        self.alpha = alpha
        self.mode = mode


        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")

        self.generate_negatives = generate_negatives(mode = mode, n_pairs = 1)

    def compute_loss(self, anchors, positives, negative_pool):
        

        negatives = tf.squeeze(self.generate_negatives(anchors, negative_pool), axis = 1)

        anchor_pos = self.distance(anchors, positives)
        anchor_neg = self.distance(anchors, negatives)
        
        loss = tf.reduce_sum(tf.math.maximum( anchor_pos - anchor_neg+ self.alpha, 0 ) )

        return loss


class centroid_loss(ContrastiveLoss):
    
    def __init__(self, n_pairs = 20, mode = 'distance_weighted_random', distance ="L2"):
        self.mode = mode


        if   distance == "L2":
            self.distance = lambda A,P: lp_distance(A,P,2)
        elif distance== "L1":
            self.distance = lambda A,P: lp_distance(A,P,1)
        else:
            exit("Currently only supports L1 and L2 distances.")


        self.generate_negatives = generate_negatives(mode = mode, n_pairs = n_pairs)

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
            

        num = tf.reduce_sum(max_vec ** -1 * AC * NC, axis=2)
        denom = tf.reduce_sum(max_vec ** -1 * AC *PC, axis=2)

        m = tf.reduce_max(tf.stack([num, denom], axis=-1), axis=-1)

        eps = 1e-12

        loss = 1 / tf.cast(n, tf.float32) * tf.reduce_sum(tf.math.log(
            1 + tf.reduce_sum(-tf.math.exp(-2.) + tf.math.exp(num - m) / (tf.math.exp(denom - m) + eps), axis=1)))


        return loss
