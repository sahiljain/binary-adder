import tensorflow as tf

input_size = 4
hidden_1_size = 6
hidden_2_size = 4
hidden_3_size = 5
output_size = 3

# Variable to populate with input
x = tf.placeholder("float", [None, input_size])

# Variable which will be populated with output
y = tf.placeholder("float", [None, output_size])

# Variables which we will be learning
weights = {
    'h1': tf.Variable(tf.random_normal([input_size, hidden_1_size])),
    'h2': tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size])),
    'h3': tf.Variable(tf.random_normal([hidden_2_size, hidden_3_size])),
    'out': tf.Variable(tf.random_normal([hidden_3_size, output_size]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden_1_size])),
    'b2': tf.Variable(tf.random_normal([hidden_2_size])),
    'b3': tf.Variable(tf.random_normal([hidden_3_size])),
    'out': tf.Variable(tf.random_normal([output_size]))
}


# Create the model
def neural_network():
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return output_layer


prediction = neural_network()

cost = tf.reduce_mean(tf.abs(tf.sub(prediction, y)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.004).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

get_from_saved = False
train = True

# Launch the graph
with tf.Session() as sess:
    if get_from_saved:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
    else:
        sess.run(init)

    if train:
        training_epochs = 5000
        for epoch in range(training_epochs):
            batch_x, batch_y = [
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 1.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 1., 1.],
                                   [0., 1., 0., 0.],
                                   [0., 1., 0., 1.],
                                   [0., 1., 1., 0.],
                                   [0., 1., 1., 1.],
                                   [1., 0., 0., 0.],
                                   [1., 0., 0., 1.],
                                   [1., 0., 1., 0.],
                                   [1., 0., 1., 1.],
                                   [1., 1., 0., 0.],
                                   [1., 1., 0., 1.],
                                   [1., 1., 1., 0.],
                                   [1., 1., 1., 1.],
                               ], [[0., 0., 0.],
                                   [0., 0., 1.],
                                   [0., 1., 0.],
                                   [0., 1., 1.],
                                   [0., 0., 1.],
                                   [0., 1., 0.],
                                   [0., 1., 1.],
                                   [1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 1., 1.],
                                   [1., 0., 0.],
                                   [1., 0., 1.],
                                   [0., 1., 1.],
                                   [1., 0., 0.],
                                   [1., 0., 1.],
                                   [1., 1., 0.]
                                   ]
            _, avg_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("saved to %s" % save_path)

    # Test model
    print "prediction 0 + 1:", prediction.eval({x: [[0., 0., 0., 1.]]})
    print "prediction 1 + 1:", prediction.eval({x: [[0., 1., 0., 1.]]})
    print "prediction 1 + 2:", prediction.eval({x: [[0., 1., 1., 0.]]})
    print "prediction 1 + 3:", prediction.eval({x: [[0., 1., 1., 1.]]})
    print "prediction 2 + 2:", prediction.eval({x: [[1., 0., 1., 0.]]})
    print weights['h1'].eval()
    print biases['b1'].eval()
    print weights['h2'].eval()
    print weights['h3'].eval()
    print weights['out'].eval()
