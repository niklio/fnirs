import os, time # operating system dependent functionality
import numpy as np # package for scientific computing on matrixes
import pandas as pd # data analysis library
import tensorflow as tf # Google neural network software

# Recurrent neural network http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# An RNN has a loop so that information persists 
from tensorflow.python.ops import rnn, rnn_cell 

log_dir = "./logs" #

train_path ="IDOXY/ID-OXY-1-train.csv"##
test_path = "IDOXY/ID-OXY-1eva.csv"##"IDOXY/ID-OXY-1.csv"

class_labels = [b'PASSTHOUGHT',b'ADDITION', b'JUNK', b'REST']
#class_labels = [b'easy', b'hard']

# Training parameters
batch_size = 50
learning_rate = 1e-4
display_step = 1

# Network parameters
n_inputs = 52
n_steps = 40
n_hidden = 32 
n_layers = 2 # 1 layer cannot figure out nonlinearity 
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


# Given an array of classes, returns a numerical represenation
def categorical_labels_to_onehot(label_batch, class_labels=None):
    df = pd.DataFrame(label_batch, columns=['label'])
    # cast to other
    df[df['label'] != class_labels[1]] = class_labels[0]

    # everything that is not the second label becomes the first 
    print(df.groupby('label').size())

    onehot_df = pd.get_dummies(df['label'], columns=class_labels)

    for label in class_labels:
        if label not in onehot_df:
            onehot_df[label] = 0

    onehot = onehot_df.as_matrix(columns=class_labels)


    #onehot: a numerical representation 
    # these are the output vectors of the NN 
    return onehot

# For a given dataset, retrieve features and label as tensors
def read_fnirs(filename_queue):
    line_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = line_reader.read(filename_queue)
    record_defaults = [[0.0]] * n_inputs + [[""]]
    cols = tf.decode_csv(csv_row, record_defaults)
    features = tf.pack(cols[:-1]) # all but the last index
    label = cols[-1]


    return features, label

# Read the data, given a path and a batch size
# Moves up one at a time 
def input_pipeline(path, batch_size=None, n_steps=None):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(path))

    features, target = read_fnirs(filename_queue)
    # Features and targets are now Tensors https://www.tensorflow.org/versions/r0.12/api_docs/python/framework.html#Tensor
    example_sequence, label_sequence = tf.train.batch([features, target],
        batch_size=n_steps)

    example_batch, label_batch = tf.train.batch([example_sequence, label_sequence],
        batch_size=batch_size)

    # keep reading examples until there is an equal distribtion 

    return example_batch, label_batch
# 
# 
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
    return tf.matmul(output[-1], weights) + biases




# Retrieve data
x_train, y_train = shuffle_input_pipeline(train_path,
    batch_size=batch_size,
    n_steps=n_steps)


x_test, y_test = shuffle_input_pipeline(test_path,
    batch_size=batch_size,
    n_steps=n_steps)



# x_train and y_train are tensors linking up to the actual data
def run(x_train, y_train, x_test, y_test):
    # TensorFlow provides a placeholder operation that must be fed with data on execution
    # It's a template, not holding a specific value but rather a preparation for computation.
    # The computations are happening outside of Python to be efficent, so most interactions
    # are via preparations of variables and template

    # X is a feature in your model. It's a raw datapoint. There are 16 inputs in our model
    # corresponding to the various channels; however these values change over time
    # n_steps is the number of subsequent readings in our model; it's 3D input  

    print(n_steps)
    # Generally, a placeholder has a data type, a shape, and a name; None is left abstract
    x = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
    #   x can hold 

    #There is y and y_; these are our classifications. There are 2 outputs of the model
    # y_ holds specifically the data type or template of our predictions
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

    # W holds the matrix of what will be learned in the actual system. We call this with
    # a shape; we want a new dimension for every possible class 
    # and we want a new dimension for the specified number of hidden layers
    W = weight_variable([n_hidden, n_classes])

    # b is the prior probabilities of our classes
    b = bias_variable([n_classes])

    # y is the un-run specification of a recurrent neural network, which is a neural
    # network with loops in it so that it remembers state. 
    y = RNN(x, W, b)

    # cost is a cross entropy reduction algorithm which takes into consideration y and y_
    # which is predictions vs answers
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    #optimizer is a way for updating variables in order to reduce loss. Why not 
    # gradient descent? Is this an arbitrary choice? 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # a matrix of correct responses depending on difference between output and ansswer
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # The cross entropy representation of correct responses
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    print("[*] Training ...")
    with tf.Session() as sess: # initiate session 
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

            if train_accuracy > 0.9:
                break

        
        # move x_test y _test into place holders, input them into computation graph
        test_examples, test_labels = sess.run([x_test, y_test])

        # converts labels into number vectors
        test_labels = categorical_labels_to_onehot(test_labels[:,-1], class_labels=class_labels)
        
        # feeds into network and classify, pass in each of 50 samples    
        test_accuracy = accuracy.eval(feed_dict={x: test_examples, y_: test_labels})

        print(test_accuracy)


        coord.request_stop()
        coord.join(threads)


print("[*] Building the network ...")


init = tf.global_variables_initializer()

run(x_train, y_train, x_test, y_test)










