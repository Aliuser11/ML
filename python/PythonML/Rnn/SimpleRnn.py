from keras.layers import rnn
import tensorflow as tf
tf.random.set_seed(1)

rnn_layer = tf.keras.layers.SimpleRNN(
    units=2,
    use_bias=True,
    return_sequences=True)

rnn_layer.build(input_shape=rnn_layer.weights)

print('W_xh shape: ', w_xh.shape) #(5,2)
print('W_oo shape: ', w_oo.shape) #(2,2)
print('b_h shape: ', b_h.shape) #(2,)
