import tensorflow as tf
import numpy as np
#filename_queue = tf.train.string_input_producer(["/home/msam/output/train-00000-of-00001"], num_epochs=1]

# Parameters
learning_rate = 0.001
training_iters = 20
#training_iters = 10
#batch_size = 8 
batch_size = 1
display_step = 10

# Network Parameters
n_input = 65536 # Going to use 256x256 of the total image
n_classes = 5 # 4 classes (arch, whorl, left loop, right loop) 5 for one hot
dropout = 1.0 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [256, 256])
y = tf.placeholder(tf.float32, [1, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string), 
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
          'image/colorspace':  tf.FixedLenFeature([], tf.string),
          'image/channels': tf.FixedLenFeature([], tf.int64),
          'image/format': tf.FixedLenFeature([], tf.string),
          'image/filename': tf.VarLenFeature(tf.string),
          'image/class/label': tf.FixedLenFeature([], tf.int64),
          'image/class/text': tf.VarLenFeature(tf.string),
       })
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
#    colorspace = features['colorspace']
    channels = tf.cast(features['image/channels'], tf.int32)
    label = tf.cast(features['image/class/label'], tf.int32)
    text = features['image/class/text']
    image_format = features['image/format']
    filename = features['image/filename']
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    print("Inside read_and_decode")
    image2 = tf.reshape(image, tf.pack([256, 256, 1]))
    #image = tf.reshape(image,tf.pack([146, 449, 3]))
    #image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
#    print("filename:", repr(filename))
#    print("height:", repr(height))
    #return image, filename, text, label, height, width
    return image2, label

def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([ FILE ], num_epochs=None, shuffle=True)
        #filename_queue = tf.train.string_input_producer([ FILE ])
        #image, filename, text, label, height, width = read_and_decode(filename_queue)
        image, label = read_and_decode(filename_queue)
        #width2 = shape1[0] / 200
        #image = tf.reshape(image, tf.pack([150, 150, 1]))
        #image2 = tf.image.resize_images(image, 200,200, 1) 
        #image.set_shape([248, 338, 3])
        init_op = tf.initialize_local_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(467):
            example, l = sess.run([image, label])
            #img = Image.fromarray(example, 'RGB')
            #img.save("output/" + str(i) + '-train.png')
            #fname,l = sess.run([filename,label])
            #print(fname,l)
            #print(example, l)
            #print(len(example))
            print(l)
            #width2 = int(len(example) / 200)
            image2 = tf.reshape(example, tf.pack([256, 256, 1]))
            print(image2.get_shape())  
            #print(h, w)

        coord.request_stop()
        coord.join(threads)



def get_examples( FILE, batch_size):
    print("Inside get_examples")
    filename_queue = tf.train.string_input_producer([ FILE ])
#    filename_queue = tf.train.string_input_producer([ FILE ], num_epochs=None, shuffle=True)
    example, label = read_and_decode(filename_queue)
    print("After read_and_decode")
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
#        init_op = tf.initialize_local_variables()
        sess.run(init)
        print("Before assigning label")
        #e, l =  sess.run([example, label])
        #print("After assigning label")
        print(sess.run(label.eval()))
#        return label
'''
        min_after_dequeue = 65536
        capacity = min_after_dequeue + batch_size * 65536
        example_batch, label_batch = tf.train.shuffle_batch(
             [example, label], batch_size=batch_size, capacity=capacity,
             min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch
'''


def get_batch_inputs(training_file, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
#  if not num_epochs: num_epochs = None
#  filename = os.path.join(FLAGS.train_dir,
#                          TRAIN_FILE if train else VALIDATION_FILE)

#  with tf.name_scope('input'):
  filename_queue = tf.train.string_input_producer(
        [training_file], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
  image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
  images, sparse_labels = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, num_threads=1,
    capacity=70000 + 3 * batch_size,
    min_after_dequeue=70000)

  print("After shuffle batch call.") 
        # Ensures a minimum amount of shuffling of examples.

  return images, sparse_labels


#batch_x, batch_y = get_batch_inputs('/home/msam/records/train_newdb_binary', 50, 1)
#print('Before get_examples')
#batch_x, batch_y = get_examples('/home/msam/records/train_newdb_binary', 2)
#get_examples('/home/msam/records/train_newdb_binary', 2)
#print("label: ", repr(label))
#print("batch_y: " , repr(batch_y))



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')


    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = avgpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = avgpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wd1': tf.Variable(tf.random_normal([64*64*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(['/home/msam/records/train_newdb_binary'])
    image, label = read_and_decode(filename_queue)
    val_filename_queue = tf.train.string_input_producer(['/home/msam/records/validation_newdb_binary'])
    val_image, val_label = read_and_decode(val_filename_queue)
    #batch_val_images, batch_val_labels = get_batch_inputs('/home/msam/records/validation_newdb_binary', 3, 1)
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
#        print("Before optimise call.")

        #example, label = get_all_records('/home/msam/records/train_bmp')
#        batch_train_images, batch_train_labels = get_batch_inputs('/home/msam/records/train_newdb_binary', 50, 1)
#        print("After get_batch_inputs call")
        #batch_x, batch_y = sess.run([batch_train_images, batch_train_labels])
#        print(batch_train_labels.eval())
#        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                       keep_prob: dropout})
#        if step % display_step == 0:
            # Calculate batch loss and accuracy
#            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                              y: batch_y,
#                                                              keep_prob: 1.0})
#
#            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))

#        print("step:" + str(step))
#        step += 1

#    coord.request_stop()
#    coord.join(threads)

#    print("Optimization Finished!")

        acc_1000 = tf.placeholder(tf.float32)
        acc_1000 = 0.0

        for i in range(1000):
            #print('in loop')
            example, l = sess.run([image, label])
            #print(l)
            example_r = np.reshape(example, (256, 256))
            one_hot_l = tf.nn.embedding_lookup(np.identity(5), l)
            op_label = (one_hot_l.eval())
            op_label = np.reshape(op_label, (-1, 5))
            #print(op_label)
            #l = np.reshape(l, [-1, 1])
            # Run optimization op (backprop)

            sess.run(optimizer, feed_dict={x: example_r, y: op_label,
                                       keep_prob: dropout})
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: example_r,
                                                              y: op_label,
                                                              keep_prob: 1.})
            if acc != 0.0:
                acc_1000 += 1.0

            #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #      "{:.5f}".format(acc))


        #imgs, labels = batch_x.eval(), batch_y.eval()

        #print("label: ", labels)
        acc_1000 /= 1000
        if(step % 5 == 0):
            print("accuracy: " + "{:.5f}".format(acc_1000))
        step += 1

        for i in range(26):
            val_x, val_y = sess.run([val_image, val_label])
            val_x_r = np.reshape(val_x, (256, 256))
            one_hot_val_l = tf.nn.embedding_lookup(np.identity(5), val_y)
            val_op_label = (one_hot_val_l.eval())
            val_op_label = np.reshape(val_op_label, (-1, 5))
            print(val_op_label)

            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: val_x_r,
                                      y: val_op_label,
                                      keep_prob: 1.}))

        coord.request_stop()
        coord.join(threads)

        # Calculate accuracy 3 test images
        #sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                              y: mnist.test.labels[:256],
        #                              keep_prob: 1.}))
        #batch_x, batch_y = get_examples('/home/msam/records/train_bmp', 1)



        #batch_x, batch_y = mnist.train.next_batch(batch_size)


        # Run optimization op (backprop)
        #sess.run(optimizer, feed_dict={x: imgs, y: labels,
        #                               keep_prob: dropout})
        #if step % display_step == 0:
            # Calculate batch loss and accuracy
        #    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
        #                                                      y: batch_y,
        #                                                      keep_prob: 1.})
        #    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
        #          "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #          "{:.5f}".format(acc))
        print("Optimization Finished!")

    # Calculate accuracy 6 test images
    #val_batch_x, val_batch_y = get_examples('/home/msam/records/val_bmp', 3)
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: val_batch_x,
    #                                  y: val_batch_y,
    #                                  keep_prob: 1.}))
        #sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                              y: mnist.test.labels[:256],
        #                              keep_prob: 1.}))

