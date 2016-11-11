import os, time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.ops import rnn, rnn_cell

log_dir = "./logs"

train_path = "./data/bestemusic/test/00.csv"
test_path = "./data/bestemusic/test/00.csv"

class_labels = [b'easy', b'hard']

# Training parameters
batch_size = 100
learning_rate = 1e-4
display_step = 1

# Network parameters
n_inputs = 16
n_steps = 100
n_hidden = 32
n_layers = 2
n_classes = len(class_labels)

print(("\n" +
"fNIRS long-short term memory network\n" +
"====================================\n" +
"batch_size={batch_size}\n" +
"learning_rate={learning_rate}\n" +
"reccurent_steps={n_steps}\n" +
"hidden_units={n_hidden}\n" +
"layers={n_layers}\n").format(
    batch_size=batch_size,
    learning_rate=learning_rate,
    n_steps=n_steps,
    n_hidden=n_hidden,
    n_layers=n_layers))

def categorical_labels_to_onehot(label_batch, class_labels=None):
    df = pd.DataFrame(label_batch, columns=['label'])
    df[df['label'] != class_labels[1]] = class_labels[0]
    print(df.groupby('label').size())
    onehot_df = pd.get_dummies(df['label'], columns=class_labels)

    for label in class_labels:
        if label not in onehot_df:
            onehot_df[label] = 0

    onehot = onehot_df.as_matrix(columns=class_labels)
    return onehot

def read_fnirs(filename_queue):
    line_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = line_reader.read(filename_queue)
    record_defaults = [[0.0]] * n_inputs + [[""]]
    cols = tf.decode_csv(csv_row, record_defaults)
    
    features = tf.pack(cols[:-1])
    label = cols[-1]
    return features, label

def input_pipeline(path, batch_size=None, n_steps=None):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(path))

    features, target = read_fnirs(filename_queue)
    example_sequence, label_sequence = tf.train.batch([features, target],
        batch_size=n_steps)

    example_batch, label_batch = tf.train.batch([example_sequence, label_sequence],
        batch_size=batch_size)

    return example_batch, label_batch

def shuffle_input_pipeline(path, batch_size=None, n_steps=None, min_after_dequeue=1000):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(path),
        shuffle=True)

    features, target = read_fnirs(filename_queue)
    example_sequence, label_sequence = tf.train.batch([features, target],
        batch_size=n_steps)

    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example_sequence, label_sequence],
        batch_size=batch_size,
        min_after_dequeue=min_after_dequeue,
        capacity=capacity)

    return example_batch, label_batch


def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def RNN(x, weights, biases, is_train=True):
    # Input shape: (batch_size, n_steps, n_input)
    x = tf.contrib.layers.batch_norm(x,
        is_training=is_train)

    # Reshape to list of n_steps tensors with shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_inputs])
    x = tf.split(0, n_steps, x)

    lstm_cell = rnn_cell.GRUCell(n_hidden, activation=tf.nn.relu)

    stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * n_layers,
        state_is_tuple=True)

    output, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)
    return tf.matmul(output[-1], W) + b


print("[*] Building the network ...")
x = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

W = weight_variable([n_hidden, n_classes])
b = bias_variable([n_classes])

y = RNN(x, W, b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train, y_train = input_pipeline(train_path,
    batch_size=batch_size,
    n_steps=n_steps)

init = tf.initialize_all_variables()

print("[*] Training ...")
with tf.Session() as sess:

    train_writer = tf.train.SummaryWriter(
        os.path.join(log_dir, "train"),
        sess.graph)

    test_writer = tf.train.SummaryWriter(
        os.path.join(log_dir, "test"),
        sess.graph)

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    step = 0
    start_time = time.time()
    while not coord.should_stop():
        examples, labels = sess.run([x_train, y_train])
        labels = categorical_labels_to_onehot(labels[:,-1], class_labels=class_labels)
        train_accuracy = accuracy.eval(feed_dict={x: examples, y_: labels})
        train_loss = cost.eval(feed_dict={x: examples, y_: labels})

        train_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy", simple_value=train_accuracy.item()),
            tf.Summary.Value(tag="Loss", simple_value=train_loss.item())])

        train_writer.add_summary(train_summary, step)

        if not step % display_step:
            print(("step={step}, time={time:.2f}, " +
                "acc={acc:.2f}, loss={loss:.2f}").format(
                    step=step,
                    time=time.time() - start_time,
                    acc=train_accuracy,
                    loss=train_loss))

        optimizer.run(feed_dict={x: examples, y_: labels})
        step += 1

    coord.request_stop()
    coord.join(threads)
