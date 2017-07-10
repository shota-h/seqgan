import os,sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, pooling
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
import keras.backend as K


args=sys.argv
lstm_num=int(args[1])
cell_num=int(args[2])
epoch_num=int(args[3])
file_dir=os.path.abspath(os.path.dirname(__file__))
signal_len=100
state_num=20
signal_num=1
EPSILON = 1e-07
class Generator():
    def __init__(self):
        self.model=self.build_network()
        self.model.summary()

    def build_network(self):
        model = Sequential()
        if lstm_num == 1:
        else:
            model.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,return_sequences=False),stateful=True)
            model.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,return_sequences=True),stateful=True)
            for i in range(lstm_num-1):
                model.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True))
            model.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=False))
        model.add(Dense(units=1,activation='sigmoid'))

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
    def __init__(self, state_n=state_num):
        self.state_n = state_n
        self.gen = Generator()
        self.disc = Discriminator()

        self.x = tf.placeholder(tf.float32,[None,signal_len,1])
        self.z = tf.placeholder(tf.float32,[None,None,1])
        self.input_buff = self.z
        # self.x_= self.z
        self.x_ = self.gen.model(self.z)
        self.x_= tf.concat([self.z,self.x_[:,:,None]],1)
        self.state_n += 1
        if self.state_n != signal_len:
            # random_state=create_random_state(int(self.z.shape[0]),s_num=signal_len-int(self.z.shape[1]))
            # self.x_= tf.concat([self.x_,random_state,1)
            for i in range(signal_len-self.state_n):
                # buff_g=self.gen.model(self.input_buff)
                buff_g=self.gen.model(self.x_)
                self.x_= tf.concat([self.x_,buff_g[:,:,None]],1)
                self.input_buff = tf.concat([self.input_buff[:,1:,:],buff_g[:,:,None]],1)

        self.d = self.disc.model(self.x)
        self.d_ = self.disc.model(self.x_)

        self.d_loss = tf.reduce_mean(-1*tf.log(self.d)-1*tf.log(1-self.d_))
        self.g_loss = tf.reduce_mean(tf.log(1-self.d_))

        self.d_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.d_loss,var_list=self.disc.model.trainable_weights)
        self.g_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.g_loss,var_list=self.gen.model.trainable_weights)

        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.r_x = create_random_state(signal_n=signal_num).tolist()
        a1=self.sess.run(self.x_, feed_dict={self.z:self.r_x})
        plt.plot(a1[0,:,0])

    def train(self,train_x):

        for i in range(epoch_num):
            # old_w=self.gen.model.get_weights()
            if (i+1) % 10 == 0:
                print('epoch{0}'.format(i+1))
            self.sess.run(self.g_opt, feed_dict = {self.z:create_random_state(signal_n=signal_num).tolist()})
            self.sess.run(self.d_opt, feed_dict = {self.x:train_x, self.z:create_random_state(signal_n=signal_num).tolist()})
            # print(np.array(old_w)-np.array(self.gen.model.get_weights()))

        a2=self.sess.run(self.x_, feed_dict={self.z:self.r_x})
        plt.plot(a2[0,:,0])

    def pre(self,train_x):
        return self.sess.run(self.d_, feed_dict = {self.z:create_random_state(signal_n=1).tolist()}),self.sess.run(self.d, feed_dict = {self.x:train_x})

def create_random_state(signal_n,state_n=state_num):
    return np.random.uniform(low=0,high=1,size=[signal_n,state_n,1])


def main():
    train_signal=np.load(file_dir+'/dataset/ecg_five_mini.npy')
    gan=GAN()
    gan.train(train_signal.tolist())
    # K.set_session(sess)
    # sess.run(tf.global_variables_initializer())
    plt.show()

    print(gan.pre(train_signal.tolist()))

    K.clear_session()

if __name__=='__main__':
    assert len(args) == 4, 'len(argv) == {0}'.format(len(args))
    main()
