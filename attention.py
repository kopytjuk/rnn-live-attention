from keras import backend as K
from keras.layers import Layer, Dense
from keras.layers.core import Permute, Reshape

import tensorflow as tf

class CustomAttentionLayer(Layer):

    def __init__(self, output_dim, activation, attention_window, score_fn, **kwargs):
        self.output_dim = output_dim
        self.attention_window = attention_window
        self.score_fn = score_fn
        self.activation = activation
        super(CustomAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.sequence_length = input_shape[-2]
        # Create a trainable weight variable for this layer.
        self.W_c = self.add_weight(name='W_c', 
                                      shape=(self.hidden_dim*2, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # x is of shape (B, T, H) # from previous layer

        # idea:
        # 1. pad the beginning of the state tensor to window size in TS dimension
        # 2. run the routines below to get an output for each timestamp
        # 3. concatenate the results to final output vector

        h = tf.pad(x, [[0, 0], [self.attention_window, 0], [0, 0]])

        out_list = list()

        for i in range(self.sequence_length):

            current_ts = self.attention_window+i
            h_t = h[:, current_ts, :] 
            h_t = tf.reshape(h_t, shape=(tf.shape(x)[0], 1, self.hidden_dim)) # B x 1 x H

            h_s = h[:, (current_ts-self.attention_window):current_ts, :] # B x (T-1) x H
        
            #h_t = K.tile(h_t, (1, K.int_shape(h_s)[1], 1))
            # desired to be B x (T-1) x 1

            if self.score_fn == "dot":
                ht_mul_hs = tf.multiply(h_t, h_s) # normal dot product
                score = K.sum(ht_mul_hs, axis=-1) # should be B x (T-1)
            else:
                ht_mul_hs = tf.multiply(h_t, h_s) # normal dot product
                score = K.sum(ht_mul_hs, axis=-1) # should be B x (T-1)

            # should be B x (T-1)
            alpha_t = K.softmax(score)

            alpha_t = tf.reshape(alpha_t, (tf.shape(h_s)[0], tf.shape(h_s)[1], 1))

            alpha_t = tf.tile(alpha_t, (1, 1, self.hidden_dim))
            hs_mul_alpha_t = tf.multiply(alpha_t, h_s)

            c_t = K.sum(hs_mul_alpha_t, axis=1)

            h_t = tf.reshape(h_t, (-1, self.hidden_dim))

            concat_vec = K.concatenate([c_t, h_t], axis=-1)

            out = K.dot(concat_vec, self.W_c)

            out = K.sigmoid(out)

            out = tf.reshape(out, shape=(-1, 1, self.output_dim))

            out_list.append(out)

        out_all = tf.concat(out_list, axis=1)

        out_all = tf.reshape(out_all, shape=(-1, self.sequence_length, self.output_dim))

        return out_all

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sequence_length ,self.output_dim)