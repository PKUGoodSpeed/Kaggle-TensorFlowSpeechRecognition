# input, output and command line tools
import os
from os.path import isdir, join
import pandas as pd

# math and data handler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# audio file i/o
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Progress bar
from progress import ProgressBar
pbar = ProgressBar()

mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

# Shuffle data
from sklearn.utils import shuffle

# Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU

hyper_pwr = 0.5
hyper_train_ratio = 0.9
hyper_n = 25
hyper_m = 15
hyper_NR = 208
hyper_NC = 112
hyper_delta = 0.5
hyper_dropout0 = 0.17
hyper_dropout1 = 0.36
hyper_dropout2 = 0.48
hyper_dropout3 = 0.64
hyper_dropout4 = 0.56
hyper_dropout5 = 0.7

TAGET_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

## Function for loading the audio data, return a dataFrame
def load_audio_data(path, ltoi):
    '''
    path: audio file path
    return: pd.DataFrame
    '''
    raw = {'x': [], 'y': [], 'label':[]}
    for i, folder in enumerate(os.listdir(path)):
        for filename in os.listdir(path + '/' + folder):
            rate, sample = wavfile.read(data_dir + '/' + folder + '/' + filename)
            assert(rate == 16000)
            if folder == 'silence':
                length = len(sample)
                for j in range(int(length/rate)):
                    raw['x'].append(np.array(sample[j*rate: (j+1)*rate]))
                    raw['y'].append(ltoi['silence'])
                    raw['label'].append('silence')
            else:
                p = max(0, rate - len(sample))
                sample = np.pad(sample, [(0,p)], mode='constant')
                sample = sample[:rate]
                raw['x'].append(np.array(sample))
                label = folder
                if folder not in TAGET_LABELS:
                    label = 'unknown'
                raw['y'].append(ltoi[label])
                raw['label'].append(label)
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
    for i in set(df.y.tolist()):
        tmp_df = df[df.y == i]
        tmp_df = shuffle(tmp_df)
        tmp_n = int(len(tmp_df)*ratio)
        train_x += tmp_df.x.tolist()[: tmp_n]
        test_x += tmp_df.x.tolist()[tmp_n: ]
        train_y += tmp_df.y.tolist()[: tmp_n]
        test_y += tmp_df.y.tolist()[tmp_n: ]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# Using fft to convert input x's
def fft_convert(samples, rate = 16000, n = 25, m = 16, NR = 256, NC = 128, delta = 1.E-10):
    '''
    convert input data into a big spectrum matrix
    '''
    res = []
    pbar.setBar(len(samples))
    for i,sam in enumerate(samples):
        pbar.show(i)
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
    
# Get Prediction
def getPrediction(model, path, mp):
    files = os.listdir(path)
    files.sort()
    dic = {'fname':[], 'label':[], 'prob':[] }
    batch_size = 10000
    y = []
    pro = []
    N = len(files)
    for i in range(0, N, batch_size):
        fnames = files[i: min(i+batch_size, N)]
        x = []
        for f in fnames:
            rate, sample = wavfile.read(path + '/' + f)
            assert(rate == 16000)
            p = max(0, rate - len(sample))
            sample = np.pad(sample, [(0,p)], mode='constant')
            sample = sample[:rate]
            x.append(sample)
        x = fft_convert(x, rate = 16000, n = hyper_n, m = hyper_m, 
        NR = hyper_NR, NC = hyper_NC, delta = hyper_delta)
        nx, ny, nz = np.shape(x)
        x = x.reshape(nx, ny, nz, 1)
        ty = model.predict_classes(x, batch_size=128)
        sy = model.predict(x, batch_size=128)
        for j,p in enumerate(ty):
            y.append(mp[p])
            pro.append(1.*sy[j][p]/np.sum(sy[j]))
        print "\n\n"
    dic['fname'] = files
    dic['label'] = y
    dic['prob'] = pro
    df = pd.DataFrame(dic)
    return df


if __name__ == '__main__':
    print "WORK STATED!!:"
    data_dir = '../data/train/audio'
    ## change the name of `_background_noise_' into 'silence` which is a proper label name
    if os.path.exists(data_dir + '/' + '_background_noise_'):
        os.system('mv {0}/_background_noise_ {1}/silence'.format(data_dir, data_dir))
    if os.path.exists(data_dir + '/' + 'silence/README.md'):
        os.system('rm {0}/silence/README.md'.format(data_dir))
    
    ## Loading raw data Frame
    print("LOADING RAW DATA!")
    label2idx = {}
    idmap = {}
    for i,lab in enumerate(TAGET_LABELS):
        label2idx[lab] = i
        idmap[i] = lab
    raw_df = load_audio_data(data_dir, label2idx)
    print raw_df[:30]
    print label2idx
    print idmap
    
    ## Parsing the data Frame into train and test sets
    print("SPLITTING DATA INTO TRAIN AND TEST SETS!")
    tr_x, tr_y, ts_x, ts_y = train_test_split(raw_df, ratio=hyper_train_ratio)
    
    ## Preprocessing x data
    print("PROCESSING FFT!")
    train_x = fft_convert(tr_x, rate = 16000, n = hyper_n, m = hyper_m, 
    NR = hyper_NR, NC = hyper_NC, delta = hyper_delta)
    test_x = fft_convert(ts_x, rate = 16000, n = hyper_n, m = hyper_m, 
    NR = hyper_NR, NC = hyper_NC, delta = hyper_delta)
    img_r, img_c = np.shape(train_x)[1:]
    train_x = train_x.reshape(len(train_x), img_r, img_c, 1)
    test_x = test_x.reshape(len(test_x), img_r, img_c, 1)
    
    ## Compute class weights
    cls_wts = comp_cls_wts(tr_y, pwr = hyper_pwr)
    print cls_wts
    
    ## Preprocessing y data
    n_cls = len(TAGET_LABELS)
    train_y = np_utils.to_categorical(tr_y, n_cls)
    test_y = np_utils.to_categorical(ts_y, n_cls)
    print("INPUT SHAPES:")
    print("train_x: ", np.shape(train_x))
    print("train_y: ", np.shape(train_y))
    print("test_x: ", np.shape(test_x))
    print("test_y: ", np.shape(test_y))
    
    ### Construct the model
    print("CONSTRUCTING MODEL!")
    model = Sequential()
    model.add(MaxPooling2D(pool_size = (2, 2), input_shape = (img_r, img_c, 1)))
    
    model.add(Conv2D(1, kernel_size = (9, 9), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(hyper_dropout1))
    
    model.add(Conv2D(2, kernel_size = (7, 7), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(hyper_dropout2))
    
    model.add(Conv2D(4, kernel_size = (5, 5), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(hyper_dropout3))
    
    model.add(Conv2D(6, kernel_size = (3, 3), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(LeakyReLU(alpha=0.005))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(hyper_dropout4))
    
    model.add(Flatten())
    
    model.add(Dense(12))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hyper_dropout5))
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()
    
    ''' First training section '''
    ### Compile the model
    N_epoch = 300
    learning_rate = 0.01
    decay_rate = 0.01
    momentum = 0.
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ### Train the model
    print("TRAINING BEGINS!")
    res = model.fit(train_x, train_y, batch_size = 128, epochs = N_epoch, 
    verbose = 1, validation_data = (test_x, test_y), 
    class_weight = cls_wts)
    print("LEARNING RATE: ", model.optimizer.lr)
    print("FIRST SECTION TRAINING ENDS!")
    train_accu = list(res.history['acc'])
    train_loss = list(res.history['loss'])
    test_accu = list(res.history['val_acc'])
    test_loss = list(res.history['val_loss'])
    
    ## Plot results
    steps = [i for i in range(len(test_accu))]
    
    statics = test_accu[200:]
    filename = "../cnn2_output/test_accu.txt"
    f = open(filename,'w')
    for acc in statics:
        f.write(str(acc) + ' ')
    f.write("\n" + str(sum(statics)*1./len(statics)))
    f.close()
    
    statics = train_accu[200:]
    filename = "../cnn2_output/train_accu.txt"
    f = open(filename,'w')
    for acc in statics:
        f.write(str(acc) + ' ')
    f.write("\n" + str(sum(statics)*1./len(statics)))
    f.close()
    
    
    print("VISUALIZATION:")
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
    
    plt.savefig('../cnn2_output/convrg_rst.png')
    
    ## show model configuration
    plot_model(model, to_file = '../cnn2_output/model.png')
    
    ## Getting prediction
    df = getPrediction(model, '../data/test/audio',idmap)
    df = df.set_index('fname')
    df.to_csv('../cnn2_output/predict.csv')
