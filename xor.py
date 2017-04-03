import tensorflow as tf

logs_path = '/tmp/tensorflow_logs/example_xor'

learning_rate = 0.015
training_epochs = 10000
display_step = 1

n_input = 2
n_hidden_1 = 3
n_classes = 1

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

with tf.name_scope('Model'):
    pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.abs(tf.sub(pred, y))) + 0.2 * (tf.reduce_sum(tf.log(tf.abs(weights['out']))) +
                                                             tf.reduce_sum(tf.log(tf.abs(weights['h1'])))
                                                             # tf.reduce_sum(tf.abs(biases['out'])) +
                                                             )

with tf.name_scope('Adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

tf.scalar_summary("loss", cost)

merged_summary_op = tf.merge_all_summaries()

saver = tf.train.Saver()

get_from_saved = True
save = False
train = True

# Launch the graph
with tf.Session() as sess:
    if get_from_saved:
        saver.restore(sess, "/tmp/model-xor.ckpt")
        print("Model restored.")
    else:
        sess.run(init)

    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    if train:
        # Training cycle
        for epoch in range(training_epochs):
            batch_x, batch_y = [
                                   [0., 0.],
                                   [0., 1.],
                                   [1., 0.],
                                   [1., 1.]
                               ], [[0.],
                                   [1.],
                                   [1.],
                                   [0.]
                                   ]
            _, avg_cost, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x,
                                                                                             y: batch_y})
            summary_writer.add_summary(summary, epoch)
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost)
        print "Optimization Finished!"
        if save:
            save_path = saver.save(sess, "/tmp/model-xor.ckpt")
            print("saved to %s" % save_path)

    print "prediction 0, 0:", pred.eval({x: [[0., 0.]]})
    print "prediction 0, 1:", pred.eval({x: [[0., 1.]]})
    print "prediction 1, 0:", pred.eval({x: [[1., 0.]]})
    print "prediction 1, 1:", pred.eval({x: [[1., 1.]]})
    print weights['h1'].eval()
    print biases['b1'].eval()
    print weights['out'].eval()
    print biases['out'].eval()
