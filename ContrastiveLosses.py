import tensorflow as tf


def gumbel_max(logits, K):
    # Inspired by (blatantly taken from) https://github.com/tensorflow/tensorflow/issues/9260
    if tf.shape(logits)[0] == 0:
        return tf.convert_to_tensor([[]], dtype=tf.int32)
    else:

        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
        _, indices = tf.nn.top_k(logits + z, K)
        return indices



def CentroidSS(anchors, positives     ):
    # This loss should have only 2D output, and no L2 normalization on the output.
    # Input shape of y_pred here is [2* batch size, dimension of embedding (most cases = 2)]

    assert len(anchors) == len(positives), f"The number of positive samples need to be the same as the number of " \
                                           f"anchors, now they have shapes {anchors.shape} vs {positives.shape} "

    # Want to have "regular" n-pair loss, but instead of computing vectors from the origin, compute
    # them with respect to some other point. We talked first about doing this with respect to the
    # mean coordinate of all the samples considered, but one thought I had was that, for large N, this
    # would in mean result in just using __approximately__ the origin.
    # Carl had an idea to use different centroids for each considered negative for each sample.
    # The centroid would then sort of be "(A+P+2*N)/4", weighting N twice since most likely A and P will lie close.

    n = tf.shape(anchors)[0] # Number of samples in the current batch.

    if tf.shape(anchors)[0] == 0:  # To not have to handle empty batches which may come up in distributed training.
        return 0.

    N_pairs = n - 1  # This means take as many as possible, There has been some issues when last batch smaller than N_pairs,
    # looping over tensors are forbidden, and is how I create one tensor, in the non-distance choice case.

    # Take at most 50 samples. This could maybe minimize the effect that we get rays from the center.
    # Do this in conjunction with distance based negatives - takes 50 closest points. If we take full batch and normalize, we will repel from the center.
    #
    N_pairs = tf.minimum(20, n - 1)

    # In the regular case, we would need to compute only one dot product for the positive samples, now
    # we need to do it N times for each sample.

    A = anchors # z[0:size // 2, :]  # Anchors
    P = positives # z[size // 2:, :]  # Positives

    A_full = tf.tile(A[:, tf.newaxis, :], [1, N_pairs, 1])
    P_full = tf.tile(P[:, tf.newaxis, :], [1, N_pairs, 1])

    # Two ways of choosing the negative samples, either as just the next sample in the batch
    # N = tf.concat([z[i + 1:n + i + 1, tf.newaxis, :] for i in range(N_pairs)], axis=1) # Negatives

    # Or sampled randomly, with the inverse distance as the unnormalized probability of being chosen
    # The Gumbel-max trick makes it possible to do categorical sampling wihout replacement.
    distances = tf.sqrt(tf.reduce_sum((A[:, :, tf.newaxis] - tf.transpose(A[:, :, tf.newaxis])) ** 2, axis=1))

    # Distance based random choice of negatives:
    logodds = tf.math.log((distances + tf.eye(tf.shape(distances)[0])) ** -1 - tf.eye(tf.shape(distances)[0]))

    # Uniform choice:
    # logodds = tf.math.log(tf.ones(tf.shape(distances)[0])  - 0.999 * tf.eye(tf.shape(distances)[0])) # 0.999 here since if == 1, then we get -inf probability. This produced error when using check_numerics

    inds = gumbel_max(logodds, N_pairs)
    N = tf.gather(A, inds)

    if tf.shape(inds)[1] == 0:
        return 0.

    C = (A_full + P_full + 2 * N) / 4  # Centroids

    # The loss function is sum_samples(log (1 + sum_negatives( exp( (A-C)'*(A-N)- (A-C)'* (A-P)  ))) )
    # Take the minus in the exponent, and compute is as a division, and correct for overflow with m and
    # underflow with eps.

    AC = A_full - C
    NC = N - C
    PC = P_full - C

    # Normalization with max_vec makes  all distances seem the same for the loss function,
    # This means that very far negatives, will now not be 0-loss, but instead aid in cluster separation.

    max_vec = tf.tile(tf.reduce_max(tf.stack(
        [tf.reduce_sum(AC ** 2, axis=-1), tf.reduce_sum(NC ** 2, axis=-1), tf.reduce_sum(PC ** 2, axis=-1)], axis=-1),
        axis=-1)[:, :, tf.newaxis], [1, 1, tf.shape(anchors)[1]])

    # max_vec =  tf.tile(tf.reduce_mean(tf.stack(
    #	[tf.reduce_sum(AC ** 2, axis=-1), tf.reduce_sum(NC ** 2, axis=-1), tf.reduce_sum(PC ** 2, axis=-1)], axis=-1), axis=-1)[:, :, tf.newaxis], [1, 1, tf.shape(z)[1]])

    # max_vec = 1

    # Changed this from a sum to a mean. I take some issue with the fact that if A=P, then AN-AP = -2, since
    # the centroid will be placed exactly in between them.
    # num = tf.reduce_mean(max_vec ** -1 * (A_full - C) * (N - C), axis=2)
    # denom = tf.reduce_mean(max_vec ** -1 * (A_full - C) * (P_full - C),axis=2)

    num = tf.reduce_sum(max_vec ** -1 * (A_full - C) * (N - C), axis=2)
    denom = tf.reduce_sum(max_vec ** -1 * (A_full - C) * (P_full - C), axis=2)

    # num = tf.cast(1. / N_pairs, tf.float32) * tf.reduce_sum((A_full - C) * (N - C), axis=2)
    # denom = tf.cast(1. / N_pairs, tf.float32) * tf.reduce_sum((A_full - C) * (P_full - C), axis=2)

    # m = tf.reduce_max(tf.stack([tf.reduce_max(num), tf.reduce_max(denom)], axis=0))

    m = tf.reduce_max(tf.stack([num, denom], axis=-1), axis=-1)

    eps = 1e-12

    loss = 1 / tf.cast(n, tf.float32) * tf.reduce_sum(tf.math.log(
        1 + tf.reduce_sum(-tf.math.exp(-2.) + tf.math.exp(num - m) / (tf.math.exp(denom - m) + eps), axis=1)))


    return loss


def Triplet(anchors, positives, alpha = 1):

    # This is just basic triplet loss

    # Try to choose the closest one in the batch as the negative?
    #anchors = z[0:size // 2, :]
    #positives = z[size // 2:, :]

    # Taking the negative as some other sample in the batch
    negatives = tf.concat([anchors[1:,:], anchors[0,tf.newaxis,:]], axis = 0)


    distances = tf.sqrt(tf.reduce_sum((anchors[:,:,tf.newaxis] - tf.transpose(anchors[:,:,tf.newaxis]))**2,axis = 1))

    argsorted_indices = tf.argsort(distances, axis = 0, direction="ASCENDING")
    #argsorted_indices[1, :]

    #if tf.random.uniform([1]) <0.0:

    #	negatives =tf.gather( anchors,argsorted_indices[5,:])
        #tf.print("nearest samples used")
    #else:
        #tf.print("just random")
    #	pass
    #tf.print(tf.shape(tf.gather( anchors,argsorted_indices[5,:])))

    # L2-distance
    anchor_pos = tf.reduce_sum( (anchors - positives) **2, axis = 1)
    anchor_neg = tf.reduce_sum( (anchors - negatives) **2 , axis = 1)

    # L1-distance
    #anchor_pos = tf.reduce_sum( tf.math.abs(anchors - positives) , axis = 1)
    #anchor_neg = tf.reduce_sum(  tf.math.abs(anchors - negatives) , axis = 1)




    return tf.reduce_sum(tf.math.maximum( anchor_pos - anchor_neg+ alpha, 0 ) )

