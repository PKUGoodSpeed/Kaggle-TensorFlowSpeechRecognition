# input, output and command line tools
import os
from os.path import isdir, join
import pandas as pd

#math and data handler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# audio file i/o
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

#Visualization
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Shuffle data
from sklearn.utils import shuffle

# Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model


## Function for loading the audio data, return a dataFrame
def load_audio_data(path):
    '''
    path: audio file path
    return: pd.DataFrame
    '''
    raw = {'x': [], 'y': [], 'label':[]}
    for i, folder in enumerate(os.listdir(path)):
        for filename in os.listdir(path + '/' + folder):
            rate, sample = wavfile.read(data_dir + '/' + folder + '/' + filename)
            assert(rate == 16000)
            raw['x'].append(np.array(sample))
            raw['y'].append(i)
            raw['label'].append(folder)
    return pd.DataFrame(raw)

# Split train, test sets, and also return label_map
def train_test_split(df, ratio = 0.7):
    '''
    return train_sets + test_sets + label_map, which maps from y to label name
    '''
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    label_map = {}
    for i in set(df.y.tolist()):
        tmp_df = df[df.y == i]
        label_map[i] = tmp_df.label.tolist()[0]
        tmp_df = shuffle(tmp_df)
        tmp_n = int(len(tmp_df)*ratio)
        train_x += tmp_df.x.tolist()[: tmp_n]
        test_x += tmp_df.x.tolist()[tmp_n: ]
        train_y += tmp_df.y.tolist()[: tmp_n]
        test_y += tmp_df.y.tolist()[tmp_n: ]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), label_map

# Using fft to convert input x's
def fft_convert(samples, rate = 16000, n = 25, m = 16, NR = 256, NC = 128, delta = 1.E-10):
    '''
    convert input data into a big spectrum matrix
    '''
    res = []
    for i,sam in enumerate(samples):
        if(i % 1000 == 0):
            print(i)
        freq, times, spec = signal.spectrogram(sam, fs=rate, window=('kaiser',10), nperseg=int(n*rate/1000),
                                               noverlap=int(m*rate/1000))
        p1 = max(0, NR - np.shape(spec)[0])
        p2 = max(0, NC - np.shape(spec)[1])
        spec = np.pad(spec, [(0,p1), (0, p2)], mode='constant')
        spec = spec[:NR, :NC]
        res.append(spec)
    return np.log(np.array(res) + delta)
    
# Function to compute class weights
def comp_cls_wts(y, pwr = 0.2):
    '''
    Used to compute class weights
    '''
    dic = {}
    for x in set(y):
        dic[x] = len(y)**pwr/list(y).count(x)**pwr
    return dic


if __name__ == '__main__':
    data_dir = '../data/train/audio'
    ## change the name of `_background_noise_' into 'silence` which is a proper label name
    if os.path.exists(data_dir + '/' + '_background_noise_'):
        os.system('mv {0}/_background_noise_ {1}/silence'.format(data_dir, data_dir))
    if os.path.exists(data_dir + '/' + 'silence/README.md'):
        os.system('rm {0}/silence/README.md'.format(data_dir))
    
    ## Loading raw data Frame
    raw_df = load_audio_data(data_dir)
    
    ## Parsing the data Frame into train and test sets
    tr_x, tr_y, ts_x, ts_y, idmap = train_test_split(raw_df, ratio=0.6)
    
    ## Preprocessing x data
    train_x = fft_convert(tr_x)
    test_x = fft_convert(ts_x)
    img_r, img_c = np.shape(train_x)[1:]
    train_x = train_x.reshape(len(train_x), img_r, img_c, 1)
    test_x = test_x.reshape(len(test_x), img_r, img_c, 1)
    
    ## Compute class weights
    cls_wts = comp_cls_wts(train_y)
    
    ## Preprocessing y data
    n_cls = 31
    train_y = np_utils.to_categorical(y_train, n_cls)
    test_y = np_utils.to_categorical(y_test, n_cls)
    
    ### Construct the model
    model = Sequential()
    model.add(MaxPooling2D(pool_size = (2, 2), input_shape = (img_r, img_c, 1)))
    model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(Itachi()))
    model.add(Dropout(0.25))
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()
    
    ### Compile the model
    optimizer = SGD()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ### Train the model
    res = model.fit(train_x, train_y, batch_size = 128, epochs = 64, 
    verbose = 1, validation_data = (test_x, test_y), 
    class_weight = cls_wts)
    
    ## Plot results
    steps = [i for i in range(128)]
    train_accu = res.history['acc']
    train_loss = res.history['loss']
    test_accu = res.history['val_acc']
    test_loss = res.history['val_loss']
    
        ## Plotting the results
    fig, axes = plt.subplots(2,2, figsize = (12, 12))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

    axes[0][0].set_title('Loss')
    axes[0][0].plot(steps, train_loss, label = 'train loss')
    axes[0][0].plot(steps, test_loss, label = 'test loss')
    axes[0][0].set_xlabel('# of steps')
    axes[0][0].legend()

    axes[0][1].set_title('Accuracy')
    axes[0][1].plot(steps, train_accu, label = 'train accuracy')
    axes[0][1].plot(steps, test_accu, label = 'test accuracy')
    axes[0][1].set_xlabel('# of steps')
    axes[0][1].legend()
    
    plt.savefig('output/convrg_rst.png')
    
    ## show model configuration
    plot_model(model, to_file = 'output/model.png')