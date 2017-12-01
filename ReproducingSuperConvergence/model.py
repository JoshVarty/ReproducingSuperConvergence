import os
import shutil
import numpy as np
import data_loader
import tensorflow as tf
from sklearn.model_selection import train_test_split

image_size = 32
num_channels = 3
pixel_depth = 255
num_labels = 10

tensorboardPath = "/tmp/svhn_single"

train_data, train_labels, test_data, test_labels, mean_image = data_loader.load_data()

print("Train data", train_data.shape, flush=True)
print("Train labels", train_labels.shape, flush=True)
print("Test data", test_data.shape, flush=True)
print("Test labels", test_labels.shape, flush=True)
print("Mean Image (training)", mean_image.shape, flush=True)

def TrainModel(min_lr, max_lr, stepsize, max_iter, name):

    def bias_variable(name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

    def weight_layer(name, shape, initializer = None):
        if initializer is None:
            #Default to initializer for Relu layers
            initializer = tf.contrib.layers.variance_scaling_initializer()

        return tf.get_variable(name, shape, 
                               initializer=initializer,
                               regularizer=tf.contrib.layers.l2_regularizer(0.0001))

    def residual_block(net, num_channels, stage, block, is_training):
        weight_name_base = 'w' + str(stage) + block + '_branch'
        bias_name_base = 'b' + str(stage) + block + '_branch'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = net

        #ConvLayer1
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2a', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2a', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2a') + bias
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95, training=is_training)
        net = tf.nn.relu(net)

        #ConvLayer2
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2b', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2b', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b') + bias
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95, training=is_training)

        #Final step: Add shortcut value to main path
        net = tf.add(net, shortcut)
        net = tf.nn.relu(net)

        return net

    def downsample_block(net, num_channels, stage, block, stride, is_training):
        weight_name_base = 'w' + str(stage) + block + '_branch'
        bias_name_base = 'b' + str(stage) + block + '_branch'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = net

        #ConvLayer1
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      2
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2a', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2a', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,2,2,1], padding='SAME', name=conv_name_base + '2a')
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95, training=is_training)
        net = tf.nn.relu(net)

        #ConvLayer2
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2b', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2b', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b')
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95, training=is_training)

        #Avg Pool (of shortcut)
        #   padding     0   ("VALID") 
        #   kernel      3x3
        #   stride      2
        shortcut = tf.nn.avg_pool(shortcut, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        
        #Add shortcut to main path
        net = tf.add(shortcut, net)
        net = tf.nn.relu(net)

        #Concatenate with zeros
        zeros = tf.zeros_like(net)
        net = tf.concat([net, zeros], axis=3)

        return net
    
    def random_flip_left_right(input):
        batch = tf.shape(input)[0]
        uniform_random = tf.random_uniform([batch], 0, 1.0)
        mirror_cond = tf.less(uniform_random, 0.5)
        result = tf.where(mirror_cond, x=input, y=tf.reverse(input, [2]))
        return result

    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name="input")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        is_training = tf.placeholder(tf.bool, name='is_training')
        mean_image_tf = tf.constant(mean_image, dtype=tf.float32)
        learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        input = tf.subtract(input, mean_image_tf)
        #If we're training, randomly flip the image
        input = tf.cond(is_training,
                                 lambda: random_flip_left_right(input),
                                 lambda: input)

        #Stage 1
        shape = input.shape.as_list()
        weights = weight_layer("w_conv1", [3, 3, shape[3], 16])
        bias = bias_variable("b_conv1", [16])
        net = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME') + bias
        net = tf.layers.batch_normalization(net, name="bn_conv1", momentum=0.95, training=is_training)
        net = tf.nn.relu(net)
        
        #Stage 2
        #RestNet Standard Block x9
        net = residual_block(net, num_channels=16, stage=2, block='a', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='b', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='c', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='d', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='e', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='f', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='g', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='h', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='i', is_training=is_training)

        #Stage3
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=16, stage=3, block='a', stride=2, is_training=is_training)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=32, stage=3, block='b', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='c', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='d', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='e', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='f', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='g', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='h', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='i', is_training=is_training)

        #Stage4
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=32, stage=4, block='a', stride=2, is_training=is_training)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=64, stage=4, block='b', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='c', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='d', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='e', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='f', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='g', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='h', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='i', is_training=is_training)

        net = tf.nn.avg_pool(net, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
        shape = net.shape.as_list()
        reshape = tf.reshape(net, [-1, shape[3]])

        #User Xavier Initialization since we're going to run this through a SoftMax
        initializer = tf.contrib.layers.xavier_initializer()
        weight = weight_layer("w_fc", [shape[3], num_labels], initializer)
        bias = bias_variable("b_fc", [num_labels])
        logits = tf.matmul(reshape, weight) + bias

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        train_prediction = tf.nn.softmax(logits)
        
        correct_prediction = tf.equal(labels, tf.cast(tf.argmax(train_prediction, 1), tf.int32))
        tf_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope(name):
            tf.summary.scalar("loss", cost)
            tf.summary.scalar("accuracy", tf_accuracy)
            tf.summary.scalar("LR", learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tensorboardPath)
            writer.add_graph(session.graph)

            #100 epochs * 400 steps each
            batch_size = 125

            tf.global_variables_initializer().run()
            for step in range(max_iter + 1):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = np.squeeze(train_labels[offset:(offset + batch_size), :])

                cycle = np.floor(1 + step / (2 * stepsize))
                x = np.abs(step/stepsize - 2 * cycle + 1)
                lr = min_lr + (max_lr - min_lr) * np.max((0.0, 1.0 - x))

                feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr, is_training: True} 

                if step % 100 == 0:
                    _, l, predictions, m, acc = session.run([optimizer, cost, train_prediction, merged, tf_accuracy], feed_dict=feed_dict)
                    writer.add_summary(m, step)

                    if step % 500 == 0:
                        print('Minibatch loss at step %d: %f' % (step, l), flush=True)
                        print('Minibatch accuracy: %.1f%%' % (acc * 100), flush=True) 
                else:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
                    
                    #If we ever end up getting NaNs, just end
                    if np.isnan(l):
                        print("Loss is NaN at step:", step)
                        break

                if step % 100 == 0:
                    #See test set performance
                    accuracySum = 0.0

                    for i in range(0, len(test_data), int(len(test_data) / 100)):
                        batch_data = test_data[i:i + int(len(test_data) / 100)]
                        batch_labels = np.squeeze(test_labels[i:i + int(len(test_data) / 100)])
                        feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr, is_training: False} 
                        l, predictions, acc = session.run([cost, train_prediction, tf_accuracy], feed_dict=feed_dict)
                        accuracySum = accuracySum + acc

                    print('Test accuracy: %.1f%%' % ((accuracySum / 100) * 100), flush=True)
        

if __name__ == '__main__':
    try:
        shutil.rmtree(tensorboardPath)
    except:
        pass

    min_lr = 0.1
    max_lr = 3.0
    stepsize = 5000
    max_iter = 10000

    TrainModel(min_lr=0.1, max_lr=3.0, stepsize=5000, max_iter=10000, name="Fig1b")
