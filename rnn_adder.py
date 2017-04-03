import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_classes = 2


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def declare_graph(num_steps=3, batch_size=16, state_size=3, learning_rate=0.1, num_inputs_per_step=4):
    reset_graph()

    x = tf.placeholder(tf.float32, [batch_size, num_steps, num_inputs_per_step], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    init_state = tf.zeros([batch_size, state_size])

    # Turn our x placeholder into a list of one-hot tensors:
    # rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
    rnn_inputs = tf.unpack(x, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

    # logits and predictions
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    # Turn our y placeholder into a list labels
    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

    loss_weights = [tf.ones([batch_size]) for _ in range(num_steps)]
    losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)

    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        saver=tf.train.Saver()
    )


def encode_output(a, length):
    ans = []
    for _ in range(length):
        a_bit = a % 2
        a /= 2
        if a_bit == 0:
            ans.append(0)
        else:
            ans.append(1)
    return np.array(ans)


def encode_input(a, b, length):
    ans = []
    for _ in range(length):
        a_bit = a % 2
        b_bit = b % 2
        a /= 2
        b /= 2
        if a_bit == 0 and b_bit == 0:
            ans.append([1, 0, 1, 0])
        elif a_bit == 0 and b_bit == 1:
            ans.append([1, 0, 0, 1])
        elif a_bit == 1 and b_bit == 0:
            ans.append([0, 1, 1, 0])
        else:
            ans.append([0, 1, 0, 1])

    return np.array(ans)


def train_network(g, num_epochs, num_digits=2, verbose=True, save=False):
    training_inputs = np.empty((0, num_digits + 1, 4))
    training_outputs = np.empty((0, num_digits + 1))
    for a in range(pow(2, num_digits)):
        for b in range(pow(2, num_digits)):
            input_sequence = encode_input(a, b, num_digits + 1)
            output_sequence = encode_output(a + b, num_digits + 1)
            training_inputs = np.vstack((training_inputs, np.array([input_sequence])))
            training_outputs = np.vstack((training_outputs, output_sequence))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for i in range(num_epochs):
            # training_state = np.zeros((training_size, state_size))
            training_state = None

            feed_dict = {g['x']: training_inputs, g['y']: training_outputs}
            if training_state is not None:
                feed_dict[g['init_state']] = training_state

            training_loss, training_state, _ = sess.run([g['total_loss'],
                                                         g['final_state'],
                                                         g['train_step']],
                                                        feed_dict)
            if verbose:
                print("Total loss at step", i,
                      ":", training_loss)
            training_losses.append(training_loss)

        if save:
            save_path = g['saver'].save(sess, "/tmp/model-rnn.ckpt")
            print("saved to %s" % save_path)

    return training_losses


def decode_output(ans):
    output = 0
    multiplier = 1
    for i in range(len(ans)):
        output += ans[i] * multiplier
        multiplier *= 2
    return output


def generate_test_output(g, num1, num2, num_digits):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, "/tmp/model-rnn.ckpt")

        state = None
        test_input = encode_input(num1, num2, num_digits)
        ans = []

        for i in range(num_digits):
            if state is not None:
                feed_dict = {g['x']: [[test_input[i]]], g['init_state']: state}
            else:
                feed_dict = {g['x']: [[test_input[i]]]}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            p = np.squeeze(preds)
            print(p)
            if p[0] > p[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans


def train():
    num_digits = 3
    graph = declare_graph(num_steps=num_digits + 1, batch_size=pow(2, num_digits*2))
    training_losses = train_network(graph, 10000, num_digits=num_digits, save=False)
    plt.plot(training_losses)
    plt.show()


def test():
    graph = declare_graph(num_steps=1, batch_size=1)

    for x in range(200, 400):
        for y in range(200, 400):
            ans = generate_test_output(graph, x, y, 10)
            sum = decode_output(ans)
            print(str(x) + " + " + str(y) + " = " + str(sum))
            print(ans)
            assert (sum == x + y)

# train()
test()
