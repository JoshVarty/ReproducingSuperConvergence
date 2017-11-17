import os
import numpy as np
import data_loader
import tensorflow as tf
from sklearn.model_selection import train_test_split

image_size = 32
num_channels = 3
pixel_depth = 255
num_labels = 10

train_data, train_labels, test_data, test_labels = data_loader.load_data()

print("Train data", train_data.shape)
print("Train labels", train_labels.shape)
print("Test data", test_data.shape)
print("Test labels", test_labels.shape)

def TrainModel(lr = 0.1):
    
    def accuracy(labels, predictions):
        return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

    def bias_variable(name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

    def weight_layer(name, shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, uniform=False))

    def residual_block(net, num_channels, stage, block):
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
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95)
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
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95)

        #Final step: Add shortcut value to main path
        net = tf.add(net, shortcut)
        net = tf.nn.relu(net)

        return net

    def downsample_block(net, num_channels, stage, block, stride=2):
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
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95)
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
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95)

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

    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name="input")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        #Stage 1
        shape = input.shape.as_list()
        weights = weight_layer("w_conv1", [3, 3, shape[3], 16])
        bias = bias_variable("b_conv1", [16])
        net = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME') + bias
        net = tf.layers.batch_normalization(net, name="bn_conv1", momentum=0.95)
        net = tf.nn.relu(net)
        
        #Stage 2
        #RestNet Standard Block x9
        net = residual_block(net, num_channels=16, stage=2, block='a')
        net = residual_block(net, num_channels=16, stage=2, block='b')
        net = residual_block(net, num_channels=16, stage=2, block='c')
        net = residual_block(net, num_channels=16, stage=2, block='d')
        net = residual_block(net, num_channels=16, stage=2, block='e')
        net = residual_block(net, num_channels=16, stage=2, block='f')
        net = residual_block(net, num_channels=16, stage=2, block='g')
        net = residual_block(net, num_channels=16, stage=2, block='h')
        net = residual_block(net, num_channels=16, stage=2, block='i')

        #Stage3
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=16, stage=3, block='a', stride=2)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=32, stage=3, block='b')
        net = residual_block(net, num_channels=32, stage=3, block='c')
        net = residual_block(net, num_channels=32, stage=3, block='d')
        net = residual_block(net, num_channels=32, stage=3, block='e')
        net = residual_block(net, num_channels=32, stage=3, block='f')
        net = residual_block(net, num_channels=32, stage=3, block='g')
        net = residual_block(net, num_channels=32, stage=3, block='h')
        net = residual_block(net, num_channels=32, stage=3, block='i')

        #Stage4
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=32, stage=4, block='a', stride=2)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=64, stage=4, block='b')
        net = residual_block(net, num_channels=64, stage=4, block='c')
        net = residual_block(net, num_channels=64, stage=4, block='d')
        net = residual_block(net, num_channels=64, stage=4, block='e')
        net = residual_block(net, num_channels=64, stage=4, block='f')
        net = residual_block(net, num_channels=64, stage=4, block='g')
        net = residual_block(net, num_channels=64, stage=4, block='h')
        net = residual_block(net, num_channels=64, stage=4, block='i')

        net = tf.nn.avg_pool(net, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
        shape = net.shape.as_list()
        reshape = tf.reshape(net, [-1, shape[3]])

        weight = weight_layer("w_fc", [shape[3], num_labels])
        bias = bias_variable("b_fc", [num_labels])
        logits = tf.matmul(reshape, weight) + bias

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar("loss_" + str(lr), cost)

        train_prediction = tf.nn.softmax(logits)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/svhn_single")
            writer.add_graph(session.graph)
            num_steps = 30000
            batch_size = 128
            tf.global_variables_initializer().run()
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = np.squeeze(train_labels[offset:(offset + batch_size), :])

                feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr} 

                if step % 500 == 0:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(batch_labels, predictions)) 

                if step % 100 == 0:
                    _, l, predictions, m = session.run([optimizer, cost, train_prediction, merged], feed_dict=feed_dict)
                    writer.add_summary(m, step)
                else:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)

            #See test set performance

            accuracySum = 0.0
            for i in range(0, len(test_data), int(len(test_data) / 10)):
                batch_data = test_data[i:i + int(len(test_data) / 10)]
                batch_labels = np.squeeze(test_labels[i:i + int(len(test_data) / 10)])
                feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr} 
                _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
                currentAccuracy = accuracy(batch_labels, predictions)
                accuracySum = accuracySum + currentAccuracy

            print('Test accuracy: %.1f%%' % (accuracySum / 10))
        

if __name__ == '__main__':
    TrainModel()
