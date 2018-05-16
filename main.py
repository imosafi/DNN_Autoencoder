import tensorflow as tf
import numpy as np
import datetime
from sklearn.utils import shuffle
from random import randint
from tensorflow.contrib.tensorboard.plugins import projector
import os
import copy

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
MNIST_DATA = 'MNIST_data'
LOG_PATH = 'logs'
STEPS = 10000
do_exteme_training = True

mnist_train_images = mnist.train.images[:50000]
mnist_train_labels = mnist.train.labels[:50000]

mnist_validation_images = np.concatenate([mnist.train.images[-5000:], mnist.validation.images])
mnist_validation_labels = np.concatenate([mnist.train.labels[-5000:], mnist.validation.labels])

mnist_test_images = mnist.test.images
mnist_test_labels = mnist.test.labels

# Parameters
fixed_learning_rate = 0.001
starter_learning_rate = 0.002
training_epochs = 100
batch_size = 100
n_input = 784

global_step = tf.Variable(0, trainable=False)
keep_prob = tf.placeholder(tf.float32)
input_layer = tf.placeholder(tf.float32, [None, n_input])


def get_nearest_neighbors_percentage(vectors, labels):
    count_matches = 0
    num_of_vectors = len(vectors)
    array = np.stack(vectors)

    two_a_b = -2 * np.matmul(array, np.transpose(array))

    temp_list = []
    for v in vectors:
        temp_list.append(np.repeat(np.dot(v,v),len(vectors)))
    x_square = np.stack(temp_list)
    y_square = np.transpose(x_square)


    calc_result = two_a_b + x_square + y_square
    np.fill_diagonal(calc_result, float("inf"))

    num_or_rows = len(calc_result)
    min_rows_indexes = np.argmin(calc_result, axis=0)

    for i in range(num_or_rows):
        if ((labels[i] == labels[min_rows_indexes[i]]).all()):
            count_matches += 1
    return str(((count_matches / num_of_vectors) * 100)) + '%'

def devide_to_batches(train_images, batch_size):
    numOfBatches = len(train_images) // batch_size # // floors the answer
    images_Batches = []
    for i in range(0, int(numOfBatches)):
        images_Batches.append((train_images[i * batch_size:i * batch_size + batch_size]))
    return np.asarray(images_Batches)

def execute_random_actions(modified_training):
    modified_set = []
    for im in modified_training:
        image = im.reshape(28, 28)

        x_shift = randint(1, 4)
        y_shift = randint(1, 4)
        x_shift_direction = randint(0, 1)
        y_shift_direction = randint(0, 1)

        if (x_shift_direction == 1):
            image = np.pad(image, ((0, 0), (x_shift, 0)), mode='constant')[:, :-x_shift]  # shift right
        else:
            image = np.pad(image, ((0, 0), (0, x_shift)), mode='constant')[:, x_shift:]  # shift left

        if (y_shift_direction == 1):
            image = np.pad(image, ((0, y_shift), (0, 0)), mode='constant')[y_shift:, :]  # shift up
        else:
            image = np.pad(image, ((y_shift, 0), (0, 0)), mode='constant')[:-y_shift, :]  # shift down

        random_row_index = randint(0, 27)  # zeroes out random row
        image[random_row_index] = 0
        random_column_index = randint(0, 27)  # zeroes out random column
        image[:, random_column_index] = 0

        modified_set.append(image.ravel())

    return modified_set

def generate_embeddings():
    sess = tf.InteractiveSession()
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(test_results[:STEPS], axis=0), trainable=False, name='embedding')

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(LOG_PATH + '/projector', sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join(LOG_PATH + '/projector/metadata.tsv')
    embed.sprite.image_path = os.path.join(MNIST_DATA + '/mnist_10k_sprite.png')
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(
        LOG_PATH, 'projector/a_model.ckpt'), global_step=STEPS)

def save_metadata(file):
    with open(file, 'w+') as f:
        for i in range(STEPS):
            c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
            f.write('{}\n'.format(c))


weights = {
    'encoder_h1': tf.Variable(tf.random_uniform([784, 256], minval=-0.1, maxval=0.1, dtype=tf.float32)),
    'encoder_h2': tf.Variable(tf.random_uniform([256, 64], minval=-0.1, maxval=0.1, dtype=tf.float32)),
    'encoder_h3': tf.Variable(tf.random_uniform([64, 30], minval=-0.1, maxval=0.1, dtype=tf.float32)),
    'decoder_h1': tf.Variable(tf.random_uniform([30, 64], minval=-0.1, maxval=0.1, dtype=tf.float32)),
    'decoder_h2': tf.Variable(tf.random_uniform([64, 256], minval=-0.1, maxval=0.1, dtype=tf.float32)),
    'decoder_h3': tf.Variable(tf.random_uniform([256, 784], minval=-0.1, maxval=0.1, dtype=tf.float32))
}

biases = {
    'encoder_b1': tf.Variable(tf.ones([256])/10),
    'encoder_b2': tf.Variable(tf.ones([64])/10),
    'encoder_b3': tf.Variable(tf.ones([30])/10),
    'decoder_b1': tf.Variable(tf.ones([64])/10),
    'decoder_b2': tf.Variable(tf.ones([256])/10),
    'decoder_b3': tf.Variable(tf.ones([784])/10)
}


#######################################

layer_1 = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['encoder_h1']), biases['encoder_b1']))
layer_1_dropout = tf.nn.dropout(layer_1, keep_prob)


layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_dropout, weights['encoder_h2']), biases['encoder_b2']))
layer_2_dropout = tf.nn.dropout(layer_2, keep_prob)


layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_dropout, weights['encoder_h3']), biases['encoder_b3']))
layer_3_dropout = tf.nn.dropout(layer_3, keep_prob)


layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3_dropout, weights['decoder_h1']), biases['decoder_b1']))
layer_4_dropout = tf.nn.dropout(layer_4, keep_prob)


layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4_dropout, weights['decoder_h2']), biases['decoder_b2']))
layer_5_dropout = tf.nn.dropout(layer_5, keep_prob)

layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5_dropout, weights['decoder_h3']), biases['decoder_b3']))

y_pred = layer_6
y_true = input_layer


learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000000, 0.96, staircase=True)

last_training_step = (tf.train.AdamOptimizer(learning_rate)
                        .minimize(tf.reduce_mean(tf.square(y_true - y_pred)), global_step=global_step))

training_step = (tf.train.AdamOptimizer(fixed_learning_rate).minimize(tf.reduce_mean(tf.square(y_true - y_pred))))

#Train
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

pretrain_time = datetime.datetime.now()
print('current time: ' + str(pretrain_time)+' start training:')


if do_exteme_training:
    for i in range(15):
        print("Extreme training, iteration " + str(i + 1))
        modified_training = execute_random_actions(copy.copy(mnist.train.images[:50000]))

        for epoch in range(training_epochs):
            shuffled_mnist_train_images = shuffle(modified_training, random_state=1)  # shuffling all the images
            train_images_batches = devide_to_batches(shuffled_mnist_train_images, batch_size)
            shuffled_train_images_batches = shuffle(train_images_batches, random_state=1)

            for batch_i in range(0, len(shuffled_train_images_batches)):
                sess.run(training_step, feed_dict={input_layer: shuffled_train_images_batches[batch_i], keep_prob: 1.0})


# Training original data
for epoch in range(training_epochs):
    shuffled_mnist_train_images = shuffle(mnist_train_images, random_state=1) # shuffling all the images
    train_images_batches = devide_to_batches(shuffled_mnist_train_images, batch_size)
    shuffled_train_images_batches = shuffle(train_images_batches, random_state=1)

    for batch_i in range(0, len(shuffled_train_images_batches)):
        sess.run(last_training_step, feed_dict={input_layer: shuffled_train_images_batches[batch_i], keep_prob: 1.0})


print('Done! training took: ' + str(datetime.datetime.now() - pretrain_time))


validation_results = sess.run(layer_3, feed_dict={input_layer:mnist_validation_images, keep_prob:1.0})
test_results = sess.run(layer_3, feed_dict={input_layer:mnist_test_images, keep_prob:1.0})

print('Validation: '+str(get_nearest_neighbors_percentage(validation_results, mnist_validation_labels)))
print('Testing: '+str(get_nearest_neighbors_percentage(test_results, mnist_test_labels)))


meta_data_path = LOG_PATH + '/projector/metadata.tsv'
print("Generate Metadata")
save_metadata(meta_data_path)
print("Generat Embeddings")
generate_embeddings()