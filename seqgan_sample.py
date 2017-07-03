import os,sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, pooling
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional


args=sys.argv
lstm_num=int(args[1])
cell_num=int(args[2])
epoch_num=int(args[3])
file_dir=os.path.abspath(os.path.dirname(__file__))
seq_len=5
signal_len=100
state_num=5
signal_num=1
EPSILON = 1e-07
class Generator():
    def __init__(self):
        self.model=self.build_network()
        self.model.summary()

    def build_network(self):
        model = Sequential()
        if (lstm_num-1) == 1:
            model.add(LSTM(input_shape=(seq_len,1),units=cell_num,unit_forget_bias=True,return_sequences=False))
        else:
            model.add(LSTM(input_shape=(seq_len,1),units=cell_num,unit_forget_bias=True,return_sequences=True))
            for i in range(lstm_num-2):
                model.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True))
            model.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=False))
        model.add(Dense(units=1))

        return model


class Discriminator():
    def __init__(self):
        self.model = self.build_network()
        self.model.summary()

    def build_network(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True),input_shape=(signal_len,1)))
        for i in range(lstm_num-1):
            model.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True)))
        model.add(Dense(units=1,activation='sigmoid'))
        model.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))

        return model


class GAN():
    def __init__(self):
        self.gen = Generator()
        self.disc = Discriminator()

        self.x = tf.placeholder(tf.float32,[None,signal_len,1])
        self.z = tf.placeholder(tf.float32,[None,state_num,1])
        self.input_buff = self.z
        self.x_= self.z
        for i in range(signal_len-state_num):
            buff_g=self.gen.model(self.input_buff)
            # print(buff_g.shape)
            # print(self.input_buff.shape)
            # print(self.x_.shape)
            self.x_= tf.concat([self.x_,buff_g[:,:,None]],1)
            self.input_buff = tf.concat([self.input_buff[:,1:,:],buff_g[:,:,None]],1)

        self.d = self.disc.model(self.x)
        self.d_ = self.disc.model(self.x_)

        self.d_loss = tf.reduce_mean(-1*tf.log(self.d)-1*tf.log(1-self.d_))
        self.g_loss = tf.reduce_mean(tf.log(1-self.d_))

        self.d_opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.d_loss,var_list=self.disc.model.trainable_weights)
        self.g_opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.g_loss,var_list=self.gen.model.trainable_weights)

    def train(self,train_x):
        self.sess.run(tf.global_variables_initializer())

        for i in range(epoch_num):
            print('epoch{0}'.format(i+1))
            self.sess.run(self.g_opt, feed_dict = {self.z:create_random_state(signal_num).tolist()})
            self.sess.run(self.d_opt, feed_dict = {self.x:train_x, self.z_:create_random_state(signal_num).tolist()})


def create_random_state(signal_num):
    return np.random.uniform(low=0,high=1,size=[signal_num,state_num,1])


def main():
    train_signal=np.load(file_dir+'/dataset/ecg_only_mini.npy')
    gan=GAN()

    with tf.Session() as sess:
        a1=sess.run(gan.d_, feed_dict={gan.z:create_random_state(signal_num).tolist()})
        gan.train(train_x.tolist())
        a2=sess.run(gan.d_, feed_dict={gan.z:create_random_state(signal_num).tolist()})

    plt.plot(a1[0,:,0])
    plt.plot(a2[0,:,0])
    plt.show()

if __name__=='__main__':
    assert len(args) == 4, 'len(argv) == {0}'.format(len(args))
    main()
