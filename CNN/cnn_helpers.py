import tensorflow as tf


#WEIGHT,BIAS INITIALIZATION
#---------------------------
#we must initialize weights with a small amount of noise for symmetry breaking and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#Since we're using ReLU , we need slighly positive bias to prevent dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



#CONVOLUTION & POOLING
#---------------------
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
