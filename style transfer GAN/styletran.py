
# coding: utf-8

# In[51]:


from __future__ import print_function, division
import scipy
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dropout, Concatenate, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import datetime
import os
from data_loader import DataLoader


# In[2]:


class styletran():
    def __init__(self):
        self.cols=256   ## image shape 
        self.rows=256
        self.nchannels=3
        self.image_shape=(self.rows, self.cols, self.nchannels)
        self.dataset='facades'  ## dataset name
        self.data_loader=DataLoader(dataset_name=self.dataset,img_res=(self.rows,self.cols))
        
        d_outshape=int(self.rows/2**4)
        self.dis_patch = (d_outshape, d_outshape, 1)
        ## start a generator
        self.generator = self.cons_G()
        imag1 = Input(shape=self.image_shape)
        imag2 = Input(shape=self.image_shape)
        test =self.generator(imag2)
        ## start a discriminator
        self.discriminator=self.cons_D()
        self.optimizer=Adam(0.0002,0.5)
        self.discriminator.compile(loss='mse',optimizer=self.optimizer,metrics=['accuracy'])
        self.discriminator.trainable = False  ## the combined model will not be put into D
        
        val=self.discriminator([test, imag2])  ## the validity of the fake image
        self.combined = Model(inputs=[imag1, imag2], outputs=[val, test])
        self.combined.compile(loss=['mse', 'mae'],loss_weights=[1, 100],optimizer=self.optimizer)
    def cons_G(self):
        def conv2d(layer_input, filters, ker_size=4, bn=True):
            ##down-sampling layers
            G1 = Conv2D(filters, kernel_size=ker_size, strides=2, padding='same')(layer_input)
            G1 = LeakyReLU(alpha=0.2)(G1)
            if bn:
                lay = BatchNormalization(momentum=0.8)(G1)
            return G1
        def deconv2d(layer_input, skip_input, filters, ker_size=4, dropout_rate=0):
            ##up-sampling layers
            G2 = UpSampling2D(size=2)(layer_input)
            G2 = Conv2D(filters, kernel_size=ker_size, strides=1, padding='same', activation='relu')(G2)
            
            if dropout_rate:
                G2 = Dropout(dropout_rate)(G2)
            G2 = BatchNormalization(momentum=0.8)(G2)
            G2 = Concatenate()([G2, skip_input])
            return G2
        ngf=64  ## n filters in the first layer of the generator
        L0=Input(shape=self.image_shape)
    ## down-sampling layers
        L1 = conv2d(L0, ngf, bn=False)
        L2 = conv2d(L1, ngf*2)
        L3 = conv2d(L2, ngf*4)
        L4 = conv2d(L3, ngf*8)
        L5 = conv2d(L4, ngf*8) 
        L6 = conv2d(L5, ngf*8)
        L7 = conv2d(L6, ngf*8)
        L8 = conv2d(L7,ngf*8)
        ##up-sampling
        l1=deconv2d(L8,L7,ngf*8)
        l2=deconv2d(l1,L6,ngf*8)
        l3=deconv2d(l2,L5,ngf*8)
        l4=deconv2d(l3,L4,ngf*8)
        l5=deconv2d(l4,L3,ngf*4)
        l6=deconv2d(l5,L2,ngf*2)
        l7=deconv2d(l6,L1,ngf)
        l8 = UpSampling2D(size=2)(l7)
        output= Conv2D(self.nchannels, kernel_size=4, strides=1, padding='same', activation='tanh')(l8)
    
        return Model(L0,output)
    def cons_D(self):
        def dislayer(layer_input, filters, ker_size=4, bn=True):
            ##Discriminator layer
            D = Conv2D(filters, kernel_size=ker_size, strides=2, padding='same')(layer_input)
            D = LeakyReLU(alpha=0.2)(D)
            if bn:
                D = BatchNormalization(momentum=0.8)(D)
            return D
        imag1=Input(shape=self.image_shape)
        imag2=Input(shape=self.image_shape)
        combined = Concatenate(axis=-1)([imag1, imag2])
        ndf=64 ## n filters of the first layer
        L1=dislayer(combined, ndf, bn=False)
        L2=dislayer(L1,ndf*2)
        L3=dislayer(L2,ndf*4)
        L4=dislayer(L3,ndf*8)
        val=Conv2D(1, kernel_size=4, strides=1, padding='same')(L4)  ## validity of the fake and real picture
        return Model([imag1, imag2], val)
    
    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()## calculate the iteration time 
        val = np.ones((batch_size,) + self.dis_patch)
        fake = np.zeros((batch_size,) + self.dis_patch)
        for epoch in range(epochs):
            for batch_i, (imag1, imag2) in enumerate(self.data_loader.load_batch(batch_size)):
                ## first train the discriminator
                test = self.generator.predict(imag2)
            
                dis_loss_r=self.discriminator.train_on_batch([imag1, imag2], val)
                dis_loss_f=self.discriminator.train_on_batch([test, imag2], fake)
                dis_loss=0.5 * np.add(dis_loss_r, dis_loss_f)
                ##train the generator
                gen_loss = self.combined.train_on_batch([imag1, imag2], [val, imag1])
                time = datetime.datetime.now() - start_time
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, 
                                                                                                       epochs,
                                                                                                       batch_i, 
                                                                                                       self.data_loader.n_batches,
                                                                                                       dis_loss[0], 
                                                                                                       100*dis_loss[1],
                                                                                                       gen_loss[0],
                                                                                                       time))
                if batch_i % sample_interval == 0:
                    os.makedirs('images/%s' % self.dataset, exist_ok=True)
                    imag1,imag2=self.data_loader.load_data(batch_size=3, is_testing=True)
                    fake_image=self.generator.predict(imag2)
                    ge_imag = np.concatenate([imag2, fake_image, imag1])
                    ge_imag = 0.5 * ge_imag + 0.5
                    titles = ['Condition', 'Generated Image', 'Original Image']
                    fig, axs = plt.subplots(3,3)
                    k=0
                    for i in range(3):
                        for j in range(3):
                            axs[i,j].imshow(ge_imag[k])
                            axs[i, j].set_title(titles[i])
                            axs[i,j].axis('off')
                            k=k+1
                    fig.savefig("images/%s/%d_%d.png" % (self.dataset, epoch, batch_i))
                    plt.close()

