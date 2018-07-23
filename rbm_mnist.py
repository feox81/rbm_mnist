#####################
## Get result path ##
#####################

import os.path
import sys

result_path = os.getcwd() + "/" + os.path.basename(sys.argv[0]).split('.')[0] + "/"
if not os.path.isdir(result_path):
    os.mkdir(result_path)




import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from utils import tile_raster_images
import enum





#################
## Const value ##
#################

visualization_per_count = 10000

learning_rate = 0.05
batchsize = 100

image_size = 28
hidden_layer_width = 10
hidden_layer_height = 10

visible_layer_size = image_size * image_size
hidden_layer_size = hidden_layer_width * hidden_layer_height

min_init_value = -0.005
max_init_value = 0.005

class Sampling(enum.Enum):
    ReLu = 0
    to_float = 1
    to_int = 2
    nothing = 3
sampling_method = Sampling.ReLu


##############
## Sampling ##
##############

def sampling_relu(probs):
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))

def sampling_float(probs):
    return tf.to_float(
    	tf.floor(
    		probs + tf.random_uniform(
    			tf.shape(probs), 
    			0, 
    			1)))

def sampling_int(probs):
    return tf.floor(
    	probs + tf.random_uniform(
    		tf.shape(probs), 
    		0, 
    		1))

def sampling_nothing(probs):
    return probs

def sampling(probs):
	if sampling_method == Sampling.ReLu:
		return sampling_relu(probs)
	elif sampling_method == Sampling.to_float:
		return sampling_float(probs)
	elif sampling_method == Sampling.to_int:
		return sampling_int(probs)
	elif sampling_method == Sampling.nothing:
		return sampling_nothing(probs)



##########################
## Variables Initialize ##
##########################

v0 				= tf.placeholder("float", [None, visible_layer_size])
weight			= tf.Variable(tf.random_uniform([visible_layer_size, hidden_layer_size], min_init_value, max_init_value))
visible_bias	= tf.Variable(tf.random_uniform([visible_layer_size],					 min_init_value, max_init_value))
hidden_bias		= tf.Variable(tf.random_uniform([hidden_layer_size],					 min_init_value, max_init_value))


#########
## RBM ##
#########

h0 = sampling_float(tf.nn.sigmoid(tf.matmul(v0, weight) + hidden_bias))
v1 = sampling_float(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(weight)) + visible_bias))
h1 = sampling_float(tf.nn.sigmoid(tf.matmul(v1, weight) + hidden_bias))

weight_positive_grad = tf.matmul(tf.transpose(v0), h0)
weight_negative_grad = tf.matmul(tf.transpose(v1), h1)



#################
## Update rule ##
#################

update_weight		= learning_rate * (weight_positive_grad - weight_negative_grad) / tf.to_float(tf.shape(v0)[0])
update_visible_bias = learning_rate * tf.reduce_mean(v0 - v1, 0)
update_hidden_bias 	= learning_rate * tf.reduce_mean(h0 - h1, 0)

update = [weight.assign_add(update_weight), 
          visible_bias.assign_add(update_visible_bias), 
          hidden_bias.assign_add(update_hidden_bias)]


##################
## Verification ##
##################

# h_sampling = sampling_float(tf.nn.sigmoid(tf.matmul(v1, weight) + hidden_bias))
# v_sampling = sampling_float(tf.nn.sigmoid(tf.matmul(h_sampling, tf.transpose(weight)) + visible_bias))
v_sampling = v1

error = v0 - v_sampling
check_error = tf.reduce_mean(error * error)



#####################
## Read MNIST Data ##
#####################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainX = mnist.train.images 
# trainY = mnist.train.labels
# testX = mnist.test.images
# testY = mnist.test.labels



#################
## Run session ##
#################

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for start, end in zip(range(0, len(trainX), batchsize), 
					  range(batchsize, len(trainX), batchsize)):

	train_x_batch = trainX[start:end]
	sess.run(update, feed_dict={v0: train_x_batch})


	###################
	## Visualization ##
	###################

	if start % visualization_per_count == 0:

		check_error_result = sess.run(check_error, feed_dict={v0: trainX})
		print("ERROR: {0}".format(check_error_result))

		hidden_layer = sess.run(h0, feed_dict={v0: trainX})
		print("HIDDEN LAYER: \n{0}".format(hidden_layer))

		train_image = Image.fromarray(tile_raster_images(X=train_x_batch,
														img_shape=(image_size, image_size),
														tile_shape=(hidden_layer_height, hidden_layer_width),
														tile_spacing=(1, 1)))
		train_image.save(result_path + "rbm_train_image_%d.png" % (start / visualization_per_count))



		weight_image = Image.fromarray(tile_raster_images(X=sess.run(weight).T,
		                                           img_shape=(image_size, image_size),
		                                           tile_shape=(hidden_layer_height, hidden_layer_width),
		                                           tile_spacing=(1, 1)))
		weight_image.save(result_path + "rbm_weight_%d.png" % (start / visualization_per_count))



		v1_image = Image.fromarray(tile_raster_images(X=sess.run(v1, feed_dict={v0: train_x_batch}),
		                                           img_shape=(image_size, image_size),
		                                           tile_shape=(hidden_layer_height, hidden_layer_width),
		                                           tile_spacing=(1, 1)))
		v1_image.save(result_path + "rbm_v1_%d.png" % (start / visualization_per_count))

		print("\n\n")




