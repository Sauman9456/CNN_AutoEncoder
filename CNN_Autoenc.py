import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from keras.callbacks import EarlyStopping

from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU)

# Loading mnist data
from keras.datasets import mnist

# input image dimensions                          
input_shape = (28, 28, 1)

# the data, shuffled and split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

'''noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)'''


#Using Gaussian distribution to add noise to the input
#‘mode=localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image.
x_train_noisy=random_noise(x_train, mode='gaussian', seed=None, clip=True)
x_test_noisy=random_noise(x_test, mode='gaussian', seed=None, clip=True)

print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')


# Implement a CNN in which the input is a noisy number and the output is a denoised number    
def CNN(features_shape, act='relu'):

    # Input
    input_layer = Input(name='inputs', shape=features_shape, dtype='float32')
    
    # Encoder
    enc_layer1 = Conv2D(32, (3, 3), activation=act, padding='same', strides=(1,1), name='en_conv1')(input_layer)
    enc_layer1 = MaxPooling2D((2, 2), strides=(2,2), padding='same', name='en_pool1')(enc_layer1)
    enc_layer2 = Conv2D(32, (3, 3), activation=act, padding='same', strides=(1,1), name='en_conv2')(enc_layer1)
    encoded = MaxPooling2D((2, 2), strides=(2,2), padding='same', name='en_pool2')(enc_layer2)
    
    # Decoder
    dec_layer1 = Conv2D(32, (3, 3), activation=act, padding='same', strides=(1,1), name='de_conv1')(encoded)
    dec_layer1 = UpSampling2D((2, 2), name='upsampling1')(dec_layer1)
    dec_layer2 = Conv2D(32, (3, 3), activation=act, padding='same', strides=(1,1), name='de_conv2')(dec_layer1)
    dec_layer2 = UpSampling2D((2, 2), name='upsampling2')(dec_layer2)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', strides=(1,1), name='de_conv3')(dec_layer2)
    
    # Printing network summary
    Model(inputs=input_layer, outputs=decoded).summary()
    
    return Model(inputs=input_layer, outputs=decoded)

batch_size = 500
epochs = 38

autoenc = CNN(input_shape, act=LeakyReLU(alpha=0.1))
autoenc.compile(optimizer='adadelta', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')

x=autoenc.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size,
            shuffle=True, validation_data=(x_test_noisy, x_test), callbacks=[early_stopping])

#Ploting training & validation loss graph
train_loss=x.history['loss']
val_loss=x.history['val_loss']
plt.figure(1,figsize=(7,5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available)
plt.style.use(['classic'])

#decoded Imges 
decoded_imgs = autoenc.predict(x_test_noisy)

# "n" is number of example digits to show
n = 10
plt.figure(figsize=(25, 5))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n/2:
    	ax.set_title('Original Images')
    # display noisy Image
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n/2:
    	ax.set_title('Noisy Input')

    # display reconstruction
    ax = plt.subplot(3, n, i+ 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n/2:
    	ax.set_title('Autoencoder Output')
plt.show()