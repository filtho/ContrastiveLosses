import sys
sys.path.append('../ContrastiveLosses')
import tensorflow as tf
import ContrastiveLosses as CL
from ContrastiveLosses import * 
import numpy as np
import inspect

def test_negatives_by_distance():

    n_pairs = 3
    coordinates = tf.cast(tf.transpose(tf.convert_to_tensor([[0,1,1,2,3,4],[1,0,2,1,2,0]])), tf.float32)
    negatives = negatives_by_distance(coordinates, coordinates, n_pairs)

    truth = tf.gather(coordinates, tf.convert_to_tensor( [[1,2,3],[0,2,3], [0,1,3], [1,2,4],[2,3,5], [1,3,4]]))
    
    for i in range(tf.shape(negatives)[0]):
        for j in range(n_pairs):
            # check that for each sample in the negatives corresponds to one of the predefined correct negatives, for all of the anchors
            assert (tf.reduce_sum( tf.cast(tf.reduce_sum(tf.cast(tf.equal(truth[i,:,:], negatives[i,j,:]), tf.int32), axis = -1) == 2, tf.int32))  == 1)



def test_number_of_pair():
    """
    This tests that the negatives that are generated are less than number of available negatives, 
    since we should not counta sample twice, and we assume that anchors \in negatives_pool, an anchors should not have itself as a negative.
    """

    anchors = tf.cast(tf.transpose(tf.convert_to_tensor([[0,1,1,2,3,4],[1,0,2,1,2,0]])), tf.float32)
    negatives_pool = tf.concat([anchors, anchors + tf.random.uniform(shape = tf.shape(anchors),minval = 0,maxval=1) ], axis = 0)
    for n_pairs in range(25):
        negatives = negatives_by_distance(anchors, negatives_pool, n_pairs)
        assert tf.shape(negatives)[1] < tf.shape(negatives_pool)[0] 

        negatives_weighted = negatives_by_distance_random(anchors, negatives_pool, n_pairs)
        assert tf.shape(negatives_weighted)[1] < tf.shape(negatives_pool)[0] 

        negatives_random = random_negatives(anchors, negatives_pool, n_pairs)
        assert tf.shape(negatives_random)[1] < tf.shape(negatives_pool)[0] 



def test_loss_overflow():
    """
    This tests whether very large input can be handled by the centroid loss function. In it, an exponential is taken, and without the m parameter, the exponential would return inf.

    """
    anchors = tf.cast(tf.transpose(tf.convert_to_tensor([[0,1,1,2,3,4],[1,0,2,1,2,0]])), tf.float32)  
    positives = anchors+ tf.random.uniform(shape= tf.shape(anchors), minval = 0, maxval = 0.1)

    CL = centroid_loss(n_pairs = 3, mode = 'distance_weighted_random', distance ="L2")

    loss = CL(anchors+1000000, positives)
    assert not tf.math.is_inf(loss) 


import matplotlib.pyplot as plt


def test_weighted_random():
    """
    This should test that, given enough runs the closest point should be most commonly chosen, and that the number of unique chosen negatives should be more than n-pairs.
    That is, we should be able to get other samples than just the n-pairs closest ones.
    """
    count = np.zeros([6,])

    for i in range(1000):
        anchors = tf.cast(tf.transpose(tf.convert_to_tensor([[0,1,1,2,3,4],[1,0,2,1,2,0]])), tf.float32)
        negatives_pool = anchors
        n_pairs = 3
        negatives_random = negatives_by_distance_random(anchors, negatives_pool, n_pairs)


        dret = np.unique(negatives_random.numpy(), axis = 1)[0]


        for x in dret:
            y = np.where(np.sum((anchors - x) == 0, axis =1)==2 )

            count[y] +=1
    #print(count)
    
    #plt.figure()
    #plt.bar(np.arange(tf.shape(anchors)[0]),count)
    #plt.savefig("test.png")

    assert count[4] != 0 and count[0] ==0 and count[1] > count[3] and count[3] > count[4]


def test_loss_basic_properties():
    """ 
    Test some basic properties of the Contrastive loss classes.
    For example, the loss should be smaller in a case where the positive is closer to the anchor, compared to the opposite case.

    This checks all the loss functions in the ContrastiveLosses module, except the base class that the others inherit from.

    """

    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    CL = clsmembers[1][1]()

    A = tf.convert_to_tensor([[0.,0.]])
    S1 = tf.convert_to_tensor([[0.,1.]])

    S2 = tf.convert_to_tensor([[2.,0.]])

    for clsmember in clsmembers:
        if clsmember[0] == "ContrastiveLoss":
            pass

        else:
            loss_function  = clsmember[1]() # Instantiate loss function
            L1 = loss_function.compute_loss(A,S1,tf.concat([A, S2], axis = 0))
            L2 = loss_function.compute_loss(A,S2,tf.concat([A, S1], axis = 0))

            assert L1 <= L2, f"Loss {clsmember[0]} failed the test"


test_loss_basic_properties()