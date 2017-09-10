import os
import tensorflow as tf
import _pickle as cPickle
import numpy
import random

#Data preprocessing
def unpickle(file):
    with open(file, 'rb') as fl:
        dict = cPickle.load(fl, encoding='bytes')

    return dict

def transform(dict):
    # return: x_train : shape [-1, 32, 32, 3] => [index_image, index_pixel_width, index_pixel_height, index_]
    x_train = []
    y_train = []
    for item in dict.items():
        key = item[0].decode('ascii')
        values = item[1]

        if key == 'data':
            x_train = values
            #x_train = numpy.reshape(values, (-1, 32, 32, 3) )

        if key == 'labels':
            y_train = values
            #y_train = numpy.array(values)

    return x_train, y_train

def get_data():
    #Get Train data
    x_train = []
    y_train = []
    for i in range(1, 6):
        data = unpickle('cifar-10-batches/data_batch_' + str(i))
        x, y = transform(data)

        x_train.append(x)
        for row in y:
            row = numpy.eye(10)[row]
            y_train.append(row)

    x_train = numpy.reshape(x_train, (-1, 32, 32, 3) ).astype('float64')
    y_train = numpy.array(y_train)

    #Get Test data
    x_test = []
    y_test = []

    data = unpickle('cifar-10-batches/test_batch')
    x, y = transform(data)

    x_test.append(x)
    for row in y:
        row = numpy.eye(10)[row]
        y_test.append(row)

    x_test = numpy.reshape(x_test, (-1, 32, 32, 3)).astype('float64')
    y_test = numpy.array(y_test)

    return x_train / 255, y_train, x_test / 255, y_test

#Easy operations
def weight_variable(shape):
    """
    Initialise weight varaibles with smalls values around 0
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Generates bias of a given shape
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """"
     Downsamples the feature map by 2X
    """
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def build_model():
    with tf.name_scope('conv1'):
        # maps 3 feature maps (channels) to 64 feature maps (conv filter size is 3x3)
        W_conv1 = weight_variable([3, 3, 3, 64]) # [index_filter_x][index_filter_y][input_size][output_size]
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        # maps 3 feature maps (channels) to 64 feature maps (conv filter size is 3x3)
        W_conv2 = weight_variable([3, 3, 64, 128])  # [index_filter_x][index_filter_y][input_size][output_size]
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        # maps 3 feature maps (channels) to 64 feature maps (conv filter size is 3x3)
        W_conv3 = weight_variable([3, 3, 128, 128])  # [index_filter_x][index_filter_y][input_size][output_size]
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    with tf.name_scope('reshape'):
        h_reshape = tf.reshape(h_conv3, [-1, 8 * 8 * 128])

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([8 * 8 * 128, 4096])
        b_fc1 = bias_variable([4096])

        h_fc1 = tf.nn.relu( tf.matmul(h_reshape, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([4096, 4096])
        b_fc2 = bias_variable([4096])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('readout'):
        W_readout = weight_variable([4096, 10])
        b_readout = bias_variable([10])

        y = tf.matmul(h_fc2, W_readout) + b_readout


    #Define loss & optimizers
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = y_, logits = y) )
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    #Define accuracy
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return optimizer, accuracy

def train(x_train, y_train, x_test, y_test, optimizer, accuracy):
    NUM_EPOCHS = 600
    BATCH_SIZE = 128
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter('tensor-log', sess.graph)
    tf.global_variables_initializer().run()

    #Loop through epochS
    for i in range(NUM_EPOCHS):
        # Choose randomly a batch of data
        int_start = random.randint(0, x_train.shape[0] - BATCH_SIZE)
        int_end = int_start + BATCH_SIZE
        batch = (x_train[int_start:int_end], y_train[int_start:int_end])

        # Run the training
        optimizer.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})

        # Run the accuracy
        #summary, train_accuracy = sess.run([merged, accuracy], feed_dict = {x: batch[0], y_: batch[1]})
        train_accuracy = sess.run([accuracy], feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, train accuracy %s' % (i, train_accuracy))

        if i % 10 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
            print('step %d, test accuracy %s' % (i, test_accuracy))

            #log_writer.add_summary(summary, i)


    # Final Accuracy
    test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
    print('Final test accuracy %g' % (test_accuracy))
    log_writer.flush()
    log_writer.close()

#Start InteractiveSession
x = tf.placeholder(tf.float32, [None, 32, 32, 3]) # input of shape [picture_index][pixel_width_index][pixel_height_index][channel_index]
y_ = tf.placeholder(tf.float32, [None, 10]) # output [picture_index][out_class_index]
sess = tf.InteractiveSession()

#Get the actual data
x_train, y_train, x_test, y_test = get_data()

#Get the actual model
optimizer, accuracy = build_model()

#Run the training process
train(x_train, y_train, x_test, y_test, optimizer, accuracy)
