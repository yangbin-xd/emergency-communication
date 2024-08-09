
# functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import math
import copy
import pickle
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib.font_manager import FontProperties

#--------------------------------------------------------
#
#  This is the source code for paper:
#  "MIMO Beamforming with Reduced Pilots and Radio Map
#   for Emergency Wireless Communications"
#   
#  The code is written by Bin Yang with MIT License
#  Any questions about the code, please email me via binyang_2020@163.com
#
#  If you find this project useful,
#  we would be grateful if you cite our paper!
#
#--------------------------------------------------------

CSI = np.load('data/CSI.npy') # (2000, 2, 32, 12)
[N, Nr, Nt, Nc] = CSI.shape

# normlize location to [0,1]
def normalize_loc(loc):
    '''
    input: (N, 2)
    output: (N, 2)

    '''
    loc_norm = np.empty(loc.shape)
    x_max = np.max(loc[:,0])
    x_min = np.min(loc[:,0])
    x_len = x_max - x_min
    loc_norm[:,0] = (loc[:,0] - x_min) / x_len
    y_max = np.max(loc[:,1])
    y_min = np.min(loc[:,1])
    y_len = y_max - y_min
    loc_norm[:,1] = (loc[:,1] - y_min) / y_len
    return loc_norm

# model of radio map scheme
def radio_map_model(input):
    '''
    ReLU can achieve better results but easy to fail
    input: (N, 2)
    output: (N, Nc, Nt, 1)

    '''
    x1 = layers.Dense(16, activation='relu')(input)
    x2 = layers.Dense(128, activation='relu')(x1)
    x3 = layers.Dense(Nc * Nt * 2, activation='tanh')(x2)
    x4 = layers.Reshape([Nc, Nt, 2])(x3)
    output = tf.complex(x4[:,:,:,0], x4[:,:,:,1])[:,:,:,None]
    return output

# model of pilot scheme
def pilot(input):
    '''
    input: partial precoding vector (N, Nc, Nt, 2) with zero padding
    output: complete precoding vector (N, Nc, Nt, 1)

    '''
    def ResNet(input):
        x1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input)
        # x1 = layers.LeakyReLU()(x1)
        # x1 = layers.BatchNormalization()(x1)
        x2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x1)
        # x2 = layers.LeakyReLU()(x2)
        # x2 = layers.BatchNormalization()(x2)
        x3 = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x2)
        # x3 = layers.BatchNormalization()(x3)
        x4 = layers.add([input, x3])
        output = x4
        return output
    # x1 = layers.Conv2D(2, (3,3), activation='sigmoid', padding='same')(input)
    for i in range(1):
        input = ResNet(input)
    x1 = input
    x2 = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x1)
    output = tf.complex(x2[:,:,:,0], x2[:,:,:,1])[:,:,:,None]
    return output

# model of fusion
def fusion(input):
    '''
    input: concatenated precoding vector (N, Nc, Nt, 4)
    output: fusioned precoding vector (N, Nc, Nt, 1)

    '''
    x1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input)
    x2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x1)
    x3 = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x2)
    output = tf.complex(x3[:,:,:,0], x3[:,:,:,1])[:,:,:,None]
    return output


# train
def train(x_train, y_train, x_test, y_test, model, epoch, noise):
    model.summary()
    model.compile(loss=su_loss(noise),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model/best_model.h5', monitor='val_loss', verbose=1,
        save_best_only=True, mode='min', save_weights_only=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=50, min_lr=1e-6, verbose=1)
    history = model.fit(x_train, y_train, epochs = epoch, batch_size = 128,
        verbose = 1, validation_split=0.5, callbacks=[checkpoint, reduce_lr]) # 
    model.evaluate(x_test, y_test, verbose=1)
    model = models.load_model('model/best_model.h5',
        custom_objects={'cust_loss': su_loss(noise)})
    return model, history

# customized loss function
def su_loss(noise):
    def cust_loss(H, v):
        '''
        input: H (N, Nc, Nr, Nt), v (N, Nc, Nt, 1)

        '''
        v_conj_tran = tf.transpose(v, (0,1,3,2), conjugate=True)
        power = tf.matmul(v_conj_tran, v)
        v_norm = v / tf.sqrt(power)
        Hv = tf.matmul(H, v_norm)
        Hv_conj = tf.transpose(Hv, (0,1,3,2), conjugate=True)
        Hv_gain = tf.abs(tf.matmul(Hv_conj, Hv))
        SNR = Hv_gain / noise
        rate = tf.math.log(1 + SNR)/np.log(2) # rate by Shannon formula
        rate_Nc = tf.reduce_mean(rate, axis=0) # average of subcarriers
        rate_mean = tf.reduce_mean(rate_Nc) # average of users
        loss = - rate_mean
        return loss
    return cust_loss

# normlize precoding vector to meet power constraint
def normalize_v(v_pred):
    '''
    input: (N, Nc, Nt, 1)
    output: (N, Nc, Nt, 1)

    '''
    v_conj_tran = np.transpose(v_pred, (0,1,3,2)).conjugate()
    power = np.matmul(v_conj_tran, v_pred)
    v_norm = v_pred / np.sqrt(power)
    return v_norm

# output normalization
def power_allocation(v):
    power = np.matmul(np.transpose(np.conj(v), (0,1,3,2)), v)
    power = np.sum(power.reshape(-1, 2*Nc), axis=-1).reshape(-1, 1)
    power = np.matmul(power, np.ones((1, 2*Nc)))
    power = power.reshape(-1, 2*Nc, 1, 1)
    v_norm = np.sqrt(Nc) * v / np.sqrt(power)
    return v_norm

# calcaulate spectral efficiency of all users
def cal_SE(H, v, noise):
    '''
    input: H (N, Nc, Nr, Nt), v (N, Nc, Nt, 1)

    '''
    Hv = np.matmul(H, v)
    Hv_gain = np.matmul(np.transpose(np.conj(Hv), (0,1,3,2)), Hv)
    Hv_gain = np.squeeze(np.abs(Hv_gain))
    SNR = Hv_gain / noise
    rate = np.log2(1 + SNR) # rate by Shannon formula
    rate_Nc = np.mean(rate, axis=1) # average of subcarriers
    return rate_Nc

# eliminate block points
def elimi_block(UEloc, CSI):
    '''
    input: UEloc (N, 3), CSI (N, Nr, Nt, Nc)
    output: UEloc (N_valid, 3), CSI (N_valid, Nr, Nt, Nc)

    '''
    index_valid = []
    for i in range(UEloc.shape[0]):
        if np.abs(np.sum(CSI[i,:])) > 1e-10:
            index_valid.append(i)
    UEloc = UEloc[index_valid,:]
    CSI = CSI[index_valid,:]
    return UEloc, CSI

# compute the SE of LoS and NLoS conditions
def compute(rate_Nc):
    '''
    input: rate_Nc (N_test, )

    '''
    LoS = np.load('data/LoS.npy')
    LoS = LoS[LoS != -1]
    N = LoS.shape[0]
    np.random.seed(1)
    np.random.shuffle(LoS) # (N, )
    N_test = rate_Nc.shape[0]
    LoS_test = LoS[N-N_test:N]
    LoS_num = 0
    NLoS_num = 0
    LoS_SE = 0
    NLoS_SE = 0
    for i in np.arange(N_test):
        if LoS_test[i] == 1:
            LoS_SE = LoS_SE + rate_Nc[i]
            LoS_num = LoS_num + 1
        if LoS_test[i] == 0:
            NLoS_SE = NLoS_SE + rate_Nc[i]
            NLoS_num = NLoS_num + 1
    LoS_SE = LoS_SE / LoS_num
    NLoS_SE = NLoS_SE / NLoS_num
    return LoS_SE, NLoS_SE

# interpolaion
def inter(H_noise, pilot_num):
    '''
    input: H (N, Nc, Ns, Nr, Nt) pilot_num select from [32, 24, 16, 8]

    '''
    if pilot_num == 32:
        Nc_pos = [0,3,6,9]
        Ns_pos = [2,5,8,11]
        [N, Nc, Ns, Nr, Nt] = H_noise.shape
        P1 = H_noise[:, Nc_pos[0], Ns_pos[0],  :,  0:2][:, None, None, :, :]
        P1 = P1.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P2 = H_noise[:, Nc_pos[1], Ns_pos[0],  :,  2:4][:, None, None, :, :]
        P2 = P2.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P3 = H_noise[:, Nc_pos[2], Ns_pos[0],  :,  4:6][:, None, None, :, :]
        P3 = P3.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P4 = H_noise[:, Nc_pos[3], Ns_pos[0],  :, 6:8][:, None, None, :, :]
        P4 = P4.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P5 = H_noise[:, Nc_pos[0], Ns_pos[1],  :, 8:10][:, None, None, :, :]
        P5 = P5.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P6 = H_noise[:, Nc_pos[1], Ns_pos[1],  :, 10:12][:, None, None, :, :]
        P6 = P6.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P7 = H_noise[:, Nc_pos[2], Ns_pos[1],  :, 12:14][:, None, None, :, :]
        P7 = P7.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P8 = H_noise[:, Nc_pos[3], Ns_pos[1],  :, 14:16][:, None, None, :, :]
        P8 = P8.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P9 = H_noise[:, Nc_pos[0], Ns_pos[2],  :, 16:18][:, None, None, :, :]
        P9 = P9.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P10 = H_noise[:, Nc_pos[1], Ns_pos[2], :, 18:20][:, None, None, :, :]
        P10 = P10.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P11 = H_noise[:, Nc_pos[2], Ns_pos[2], :, 20:22][:, None, None, :, :]
        P11 = P11.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P12 = H_noise[:, Nc_pos[3], Ns_pos[2], :, 22:24][:, None, None, :, :]
        P12 = P12.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P13 = H_noise[:, Nc_pos[0], Ns_pos[3], :, 24:26][:, None, None, :, :]
        P13 = P13.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P14 = H_noise[:, Nc_pos[1], Ns_pos[3], :, 26:28][:, None, None, :, :]
        P14 = P14.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P15 = H_noise[:, Nc_pos[2], Ns_pos[3], :, 28:30][:, None, None, :, :]
        P15 = P15.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P16 = H_noise[:, Nc_pos[3], Ns_pos[3], :, 30:32][:, None, None, :, :]
        P16 = P16.repeat(Nc, axis=1).repeat(Ns, axis=2)
        H_conc = np.concatenate(
            [P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16], axis=-1)
        return H_conc
    elif pilot_num == 24:
        Nc_pos = [0,2,4,6,8,10]
        Ns_pos = [2,11]
        [N, Nc, Ns, Nr, Nt] = H_noise.shape
        P1 = H_noise[:, Nc_pos[0], Ns_pos[0],  :,  0:2][:, None, None, :, :]
        P1 = P1.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P2 = H_noise[:, Nc_pos[1], Ns_pos[0],  :,  2:6:2][:, None, None, :, :]
        P2 = P2.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P3 = H_noise[:, Nc_pos[2], Ns_pos[0],  :,  5:7][:, None, None, :, :]
        P3 = P3.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P4 = H_noise[:, Nc_pos[3], Ns_pos[0],  :, 8:10][:, None, None, :, :]
        P4 = P4.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P5 = H_noise[:, Nc_pos[4], Ns_pos[0],  :, 10:14:2][:, None, None, :, :]
        P5 = P5.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P6 = H_noise[:, Nc_pos[5], Ns_pos[0],  :, 13:15][:, None, None, :, :]
        P6 = P6.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P7 = H_noise[:, Nc_pos[0], Ns_pos[1],  :, 16:18][:, None, None, :, :]
        P7 = P7.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P8 = H_noise[:, Nc_pos[1], Ns_pos[1],  :, 18:22:2][:, None, None, :, :]
        P8 = P8.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P9 = H_noise[:, Nc_pos[2], Ns_pos[1],  :, 21:23][:, None, None, :, :]
        P9 = P9.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P10 = H_noise[:, Nc_pos[3], Ns_pos[1], :, 24:26][:, None, None, :, :]
        P10 = P10.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P11 = H_noise[:, Nc_pos[4], Ns_pos[1], :, 26:30:2][:, None, None, :, :]
        P11 = P11.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P12 = H_noise[:, Nc_pos[5], Ns_pos[1], :, 29:31][:, None, None, :, :]
        P12 = P12.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P = np.zeros(P1[:,:,:,:,0][:,:,:,:,None].shape)
        H_conc = np.concatenate(
            [P1[:,:,:,:,0][:,:,:,:,None],P1[:,:,:,:,1][:,:,:,:,None],P2[:,:,:,:,0][:,:,:,:,None],P,
             P2[:,:,:,:,1][:,:,:,:,None],P3[:,:,:,:,0][:,:,:,:,None],P3[:,:,:,:,1][:,:,:,:,None],P,
             P4[:,:,:,:,0][:,:,:,:,None],P4[:,:,:,:,1][:,:,:,:,None],P5[:,:,:,:,0][:,:,:,:,None],P,
             P5[:,:,:,:,1][:,:,:,:,None],P6[:,:,:,:,0][:,:,:,:,None],P6[:,:,:,:,1][:,:,:,:,None],P,
             P7[:,:,:,:,0][:,:,:,:,None],P7[:,:,:,:,1][:,:,:,:,None],P8[:,:,:,:,0][:,:,:,:,None],P,
             P8[:,:,:,:,1][:,:,:,:,None],P9[:,:,:,:,0][:,:,:,:,None],P9[:,:,:,:,1][:,:,:,:,None],P,
             P10[:,:,:,:,0][:,:,:,:,None],P10[:,:,:,:,1][:,:,:,:,None],P11[:,:,:,:,0][:,:,:,:,None],P,
             P11[:,:,:,:,1][:,:,:,:,None],P12[:,:,:,:,0][:,:,:,:,None],P12[:,:,:,:,1][:,:,:,:,None],P], axis=-1)
        return H_conc
    elif pilot_num == 16:
        Nc_pos = [0,3,6,9]
        Ns_pos = [2,11]
        [N, Nc, Ns, Nr, Nt] = H_noise.shape
        P1 = H_noise[:, Nc_pos[0], Ns_pos[0], :,  0:4:2][:, None, None, :, :]
        P1 = P1.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P2 = H_noise[:, Nc_pos[1], Ns_pos[0], :,  4:8:2][:, None, None, :, :]
        P2 = P2.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P3 = H_noise[:, Nc_pos[2], Ns_pos[0], :,  8:12:2][:, None, None, :, :]
        P3 = P3.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P4 = H_noise[:, Nc_pos[3], Ns_pos[0], :, 12:16:2][:, None, None, :, :]
        P4 = P4.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P5 = H_noise[:, Nc_pos[0], Ns_pos[1], :, 16:20:2][:, None, None, :, :]
        P5 = P5.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P6 = H_noise[:, Nc_pos[1], Ns_pos[1], :, 20:24:2][:, None, None, :, :]
        P6 = P6.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P7 = H_noise[:, Nc_pos[2], Ns_pos[1], :, 24:28:2][:, None, None, :, :]
        P7 = P7.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P8 = H_noise[:, Nc_pos[3], Ns_pos[1], :, 28:32:2][:, None, None, :, :]
        P8 = P8.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P = np.zeros(P1[:,:,:,:,0][:,:,:,:,None].shape)
        H_conc = np.concatenate(
            [P1[:,:,:,:,0][:,:,:,:,None],P,P1[:,:,:,:,1][:,:,:,:,None],P,
             P2[:,:,:,:,0][:,:,:,:,None],P,P2[:,:,:,:,1][:,:,:,:,None],P,
             P3[:,:,:,:,0][:,:,:,:,None],P,P3[:,:,:,:,1][:,:,:,:,None],P,
             P4[:,:,:,:,0][:,:,:,:,None],P,P4[:,:,:,:,1][:,:,:,:,None],P,
             P5[:,:,:,:,0][:,:,:,:,None],P,P5[:,:,:,:,1][:,:,:,:,None],P,
             P6[:,:,:,:,0][:,:,:,:,None],P,P6[:,:,:,:,1][:,:,:,:,None],P,
             P7[:,:,:,:,0][:,:,:,:,None],P,P7[:,:,:,:,1][:,:,:,:,None],P,
             P8[:,:,:,:,0][:,:,:,:,None],P,P8[:,:,:,:,1][:,:,:,:,None],P], axis=-1)
        return H_conc
    elif pilot_num == 8:
        Nc_pos = [0,3,6,9]
        Ns_pos = [2]
        [N, Nc, Ns, Nr, Nt] = H_noise.shape
        P1 = H_noise[:, Nc_pos[0], Ns_pos[0], :,  0:8:4][:, None, None, :, :]
        P1 = P1.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P2 = H_noise[:, Nc_pos[1], Ns_pos[0], :,  8:16:4][:, None, None, :, :]
        P2 = P2.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P3 = H_noise[:, Nc_pos[2], Ns_pos[0], :, 16:24:4][:, None, None, :, :]
        P3 = P3.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P4 = H_noise[:, Nc_pos[3], Ns_pos[0], :, 24:32:4][:, None, None, :, :]
        P4 = P4.repeat(Nc, axis=1).repeat(Ns, axis=2)
        P = np.zeros(P1[:,:,:,:,0][:,:,:,:,None].shape)
        H_conc = np.concatenate(
            [P1[:,:,:,:,0][:,:,:,:,None],P,P,P,P1[:,:,:,:,1][:,:,:,:,None],P,P,P,
             P2[:,:,:,:,0][:,:,:,:,None],P,P,P,P2[:,:,:,:,1][:,:,:,:,None],P,P,P,
             P3[:,:,:,:,0][:,:,:,:,None],P,P,P,P3[:,:,:,:,1][:,:,:,:,None],P,P,P,
             P4[:,:,:,:,0][:,:,:,:,None],P,P,P,P4[:,:,:,:,1][:,:,:,:,None],P,P,P], axis=-1)
        return H_conc
    else:
        print('The pilot number is not compliant with Bin Yang\'s standards :(')
        print('Bin Yang reminds you that the selectable numbers are 32, 24, 16, and 8.')
        sys.exit()


