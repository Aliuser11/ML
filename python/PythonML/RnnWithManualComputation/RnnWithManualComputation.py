
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

x_seq = tf.convert_to_tensor(
    [[1.0]*5,[2.0]*5,[3.0]*5],
     dtype=tf.float32)

# simple rnn -> output
output = rnn_layer(tf.reshape(x_seq, shape=(1,3,5)))

# output computed manually
out_man = []

for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t],(1,5))
    print('Time step {} => '.format(t))
    print('   input     :', xt.numpy())

    ht = tf.matmul(xt,w_xh) + b_h
    print('     Hidden    :', ht.numpy())

if t>0:
    prev_o = out_man[t-1]
else:
        prev_o = tf.zeros(shape=(ht.shape))

ot=ht + tf.matmul(prev_o, w_oo)
ot=tf.math.tanh(ot)
out_man.append(ot)
print('    Output(manual)  :', ot.numpy())
print('   SimpleRnn output :' .format(t),output[0][t].numpy())
print()


