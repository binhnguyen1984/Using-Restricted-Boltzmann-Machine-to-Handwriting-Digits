import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
# load data
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0],1,28,28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0],1,28,28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0],28,28,1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0],28,28,1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images/=255
test_images/=255

train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

class RBM(object):
    def __init__(self, input_shape, hidden_units):
        self.hidden_units = hidden_units
        self.input_shape=input_shape
        self.__add_weights()
    
    def __add_weights(self):
        assert len(self.input_shape)==2
        self.W = tf.Variable(name='weights', dtype=tf.float32, initial_value=tf.zeros((self.input_shape[1], self.hidden_units)), trainable=True)
        self.v_b = tf.Variable(name='visible_bias', dtype=tf.float32, initial_value=tf.zeros([self.input_shape[1]]), trainable=True)
        self.h_b = tf.Variable(name='hidden_bias', dtype=tf.float32, initial_value=tf.zeros([self.hidden_units]), trainable=True)
        
    def __forward(self, inputs):
        # processing the input
        assert inputs.shape[1]==self.input_shape[1]
        v_state = inputs
        h_prob = tf.nn.sigmoid(tf.matmul(v_state, self.W)+ self.h_b)
        h_state = tf.nn.relu(tf.sign(h_prob-tf.random_uniform(tf.shape(h_prob))))
        return h_prob, h_state
    
    def __backward(self, h_state):
        # reconstructing the input
        v_prob = tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(self.W))+ self.v_b)
        v_state = tf.nn.relu(tf.sign(v_prob-tf.random_uniform(tf.shape(v_prob))))
        return v_prob, v_state
    
    def __one_pass(self, v0_state):
        # process one forward and backward round
        h0_prob, h0_state = self.__forward(v0_state)
        v1_prob, v1_state = self.__backward(h0_state)
        return h0_prob, h0_state, v1_prob, v1_state
            
    def fit(self, inputs, learning_rate=0.01, epochs=5, batch_size=100):
        v0_state = tf.placeholder(shape=self.input_shape, dtype=tf.float32)
        h0_prob, h0_state, v1_prob, v1_state = self.__one_pass(v0_state)
        h1_prob, h1_state, _, _ = self.__one_pass(v1_state)
        loss = tf.reduce_mean(tf.square(v0_state-v1_state))

        W_delta = tf.matmul(tf.transpose(v0_state), h0_prob) - tf.matmul(tf.transpose(v1_state), h1_prob)
        update_w = self.W.assign_add(learning_rate * W_delta)
        update_vb = self.v_b.assign_add(learning_rate * tf.reduce_mean(v0_state - v1_state, 0))
        update_hb = self.h_b.assign_add(learning_rate * tf.reduce_mean(h0_state - h1_state, 0))
        init_vars = tf.global_variables_initializer()
        
        self.weights = []
        self.errors = []
        
        with tf.Session() as sess:
            sess.run(init_vars)
            for epoch in range(epochs):
                error = []
                for start, end in zip( range(0, len(inputs), batch_size), range(batch_size, len(inputs), batch_size)):
                    batch = inputs[start:end]
                    w,_,_, err = sess.run([update_w, update_vb, update_hb, loss], feed_dict={v0_state:batch})
                    error.append(err)
                mean_err = np.mean(error)
                self.errors.append(mean_err)
                self.weights.append(w)
                print("epoch:{}, loss:{}".format(epoch, mean_err))
        
rbm = RBM(input_shape=(None, 784), hidden_units=56)
rbm.fit(train_images, learning_rate=0.01, epochs=50)

# plot the error curve
plt.plot(rbm.errors)
plt.xlabel("Batch number")
plt.ylabel("Error")
plt.show()

# one-hot encoding digit labels
train_labels = tf.keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = tf.keras.utils.to_categorical(mnist_test_labels, 10)

