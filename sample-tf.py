import tensorflow as tf

#need to import data here
#mnist data loaded here, change!
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#one_hot - a vector which is 0 in most dimensions and 1 in a single dimension.
#The dimension that corresponds to the digit that each image represents is marked as 1.
#BTW, tensor means n-dimensional array.

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

##mnist dependent data below (inputs and classes) change for other data sets
n_classes = 10          ###mnist data is handwritten digits 0-9
n_inputs = 784          ###mnist data is 28x28 pixels which we squahs to 784
batch_size = 500     ### batch size can be tweaked depending on amount of RAM

## x is your data, you do not need to specify the size of the matrix [None, n_inputs], 
## but this will cause tensorflow to throw an error if data is loaded outside of that shape
## This may need to be changed if your data set is not the mnist
x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float',[None, n_classes])
#x is a placeholder, a value we'll input when we ask tensorflow to run a computation.
#The first argument which is 'None' means that a dimension can be of any length. 
W = tf.Variable(tf.zeros([n_inputs,n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
#Outputs random values from a truncated normal distribution.
#The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#The strides argument specifies how much the window shifts in each of the input dimensions. The dimensions are batch, height, width, chanels. Here, because the image is greyscale, there is only 1 dimension. A batch size of 1 means that a single example is processed at a time. A value of 1 in row/column means that we apply the operator at every row and column. When the padding parameter takes the value of 'SAME', this means that for those elements which have the filter sticking out during convolution, the portions of the filter that stick out are assumed to contribute 0 to the dot product. The size of the image remains the same after the operation remains the same. When the padding = 'VALID' option is used, the filter is centered along the center of the image, so that the filter does not stick out. So the output image size would have been 3X3. 

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def neural_network_model(data):
    W_fc1 = weight_variable([n_inputs, n_nodes_hl1])
    b_fc1 = bias_variable([n_nodes_hl1])

    W_fc2 = weight_variable([n_nodes_hl1, n_nodes_hl2])
    b_fc2 = bias_variable([n_nodes_hl2])

    W_fc3 = weight_variable([n_nodes_hl2, n_nodes_hl3])
    b_fc3 = bias_variable([n_nodes_hl3])

    W_output = weight_variable([n_nodes_hl3,n_classes])
    b_output = bias_variable([n_classes])

    l1 = tf.nn.relu(tf.matmul(data,W_fc1) + b_fc1)
    l2 = tf.nn.relu(tf.matmul(l1,W_fc2) + b_fc2)
    l3 = tf.nn.relu(tf.matmul(l2,W_fc3) + b_fc3)

    output = tf.matmul(l3, W_output) + b_output
    return output

def conv_network_model(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
#Weight variable dimensions - filter dimensions - width x height x channels x number of filters
#Number of channels in output image - number of filters in the filter
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
#A value of -1 for the reshape parameter is to allow the input to dynamically decide what value that parameter should take.
#Since we use x_image as an input to conv2d, it would seem we are changing it from the 1X784 shape to a shape which has 4 parameters. 
#The first new parameter would be the number of images. 
#The second and third would be the dimensions of the image.
#The fourth would be the number of channels in the input image. 


    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#After convolution, the number of images would be the same as that in x_image. The dimensions would depend on image size, filter size and padding. The number of channels would depend on the number of channels in the filter.
    h_pool1 = max_pool_2x2(h_conv1)
#At the end of the first pooling layer, there is a 14X14 image.

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
#At the end of the second pooling layer, there is a 7X7 image.

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
#1024 - an arbitrary choice?

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

#    output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    output=tf.matmul(h_fc1, W_fc2) + b_fc2
    return output

def train_neural_network(x, network_model):
    prediction = network_model(x)
#    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#logit - unnormalized log of probability
#softmax - allows normalization of values
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#tf.equal - returns the truth value of whether the 2 arguments were equal.
#checks whether for the first dimension the largest probabiliry with one hot encoding is the same with the predicted and actual values. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#reduce_mean reduces all dimensions by computing the mean.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y: batch[1]})
            print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
            #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
            #train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

def other_train_neural_network(x, network_model):
    prediction = network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    train_step = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        hm_epochs = 10
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([train_step, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
#Calculates loss for the batch for reporting.
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x, neural_network_model)
train_neural_network(x, conv_network_model)
