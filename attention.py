from keras import backend as K
from keras.layers import Layer, Dense
from keras.layers.core import Permute, Reshape

import tensorflow as tf

class CustomAttentionLayer(RNN):

    def __init__(self, output_dim, activation, attention_window, score_fn, **kwargs):
        self.output_dim = output_dim
        self.attention_window = attention_window
        self.score_fn = score_fn
        self.activation = activation
        super(CustomAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.W_c = self.add_weight(name='W_c', 
                                      shape=(self.hidden_dim*2, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, states):
        # x is of shape (B, T, H) # from previous layer

        h_t = inputs[:, -1:, :] # B x 1 x H
        h_s = inputs[:, :-1, :] # B x (T-1) x H

        print("h_t", h_t)
        print("h_s", h_s)

        #h_t = K.tile(h_t, (1, K.int_shape(h_s)[1], 1))
        # desired to be B x (T-1) x 1
        ht_mul_hs = tf.multiply(h_t, h_s)
        score = K.sum(ht_mul_hs, axis=-1)

        # should be B x (T-1)
        print("score", score)

        # should be B x (T-1)
        alpha_t = K.softmax(score)

        print("alpha_t", alpha_t)

        alpha_t = tf.reshape(alpha_t, (tf.shape(h_s)[0], tf.shape(h_s)[1], 1))

        alpha_t = tf.tile(alpha_t, (1, 1, self.hidden_dim))
        hs_mul_alpha_t = tf.multiply(alpha_t, h_s)

        c_t = K.sum(hs_mul_alpha_t, axis=1)
        
        print("c_t", c_t)

        h_t = tf.reshape(h_t, (-1, self.hidden_dim))

        concat_vec = K.concatenate([c_t, h_t], axis=-1)

        print("concat_vec", concat_vec)

        out = K.dot(concat_vec, self.W_c)

        return K.sigmoid(out), [inputs]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)