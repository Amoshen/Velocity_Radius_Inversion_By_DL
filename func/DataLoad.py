import numpy as np
import skimage
import scipy.io
import os
from skimage.measure import block_reduce

def DataLoad_Train(train_size, train_data_dir, data_dim, data_dsp_blk, label_dsp_blk, position, truthfilename, velocity_flag, model_dim = [0,0]):
    for root, dirs, files in os.walk(train_data_dir+'georec_train/'):
        flag_start = True
        for file in files:
            #Load .mat data
            data_train = scipy.io.loadmat(os.path.join(root, file))
            num = file[:position[-1]]
            data_train = np.float32(data_train[str(file[:-4])].reshape([data_dim[0],data_dim[1]]))
            data_train = block_reduce(data_train, block_size = data_dsp_blk, func = decimate)
            #pick only one waveform from vsp
            data_train = data_train[-1:,:]
            data_dsp_dim   = data_train.shape

            filename_label = train_data_dir+'vmodel_train/'+num+truthfilename+'.mat'
            data_label = scipy.io.loadmat(filename_label)
            data_label = np.float32(data_label[num+truthfilename].reshape(model_dim))
            data_label = block_reduce(data_label,block_size=label_dsp_blk,func=np.max)
            data_label = data_label[-1,:].reshape(1,-1)
            radi       = [float(num[:position[0]])]
            for i in range(1,len(position)):
                radi.append(float(num[position[i-1]+1:position[i]]))
            radius     = np.array(sorted(radi),dtype=np.float32).reshape(1,-1)
            label_dsp_dim = data_label.shape
            data_label = data_label.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
            if flag_start:
                train_set      = data_train
                label_set      = data_label
                radius_set     = radius
                flag_start     = False
            else:
                train_set      = np.append(train_set,data_train,axis=0)
                label_set      = np.append(label_set,data_label,axis=0)
                radius_set     = np.append(radius_set,radius,axis=0)
    
    train_set = train_set.reshape((train_size,1,data_dsp_dim[0],data_dsp_dim[1]))
    label_set = label_set.reshape((train_size,1,label_dsp_dim[0],label_dsp_dim[1]))
    radius_set = radius_set.reshape((train_size,1,1,len(position)))
    

    if velocity_flag:
        train_set = (train_set - train_set.min())/(train_set.max() - train_set.min())
        label_set = (label_set - label_set.min())/(label_set.max() - label_set.min())
    return train_set, label_set, data_dsp_dim, label_dsp_dim, radius_set

def DataLoad_Test(test_size, test_data_dir, data_dim, data_dsp_blk, label_dsp_blk, position, truthfilename, velocity_flag, model_dim = [0,0]):
    for root, dirs, files in os.walk(test_data_dir+'georec_test/'):
        flag_start = True
        for file in files:
            #Load .mat data
            data_test = scipy.io.loadmat(os.path.join(root, file))
            num = file[:position[-1]]
            data_test = np.float32(data_test[str(file[:-4])].reshape([data_dim[0],data_dim[1]]))
            data_test = block_reduce(data_test, block_size = data_dsp_blk, func = decimate)
            #pick only one waveform from vsp
            data_test = data_test[-1:,:]
            data_dsp_dim   = data_test.shape

            filename_label = test_data_dir+'vmodel_test/'+num+truthfilename+'.mat'
            data_label = scipy.io.loadmat(filename_label)
            data_label = np.float32(data_label[num+truthfilename].reshape(model_dim))
            data_label = block_reduce(data_label,block_size=label_dsp_blk,func=np.max)
            data_label = data_label[-1,:].reshape(1,-1)
            radi       = [float(num[:position[0]])]
            for i in range(1,len(position)):
                radi.append(float(num[position[i-1]+1:position[i]]))
            radius = np.array(sorted(radi),dtype=np.float32).reshape(1,-1)
            label_dsp_dim = data_label.shape
            data_label = data_label.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
            if flag_start:
                test_set      = data_test
                label_set      = data_label
                radius_set     = radius
                flag_start     = False
            else:
                test_set      = np.append(test_set,data_test,axis=0)
                label_set      = np.append(label_set,data_label,axis=0)
                radius_set     = np.append(radius_set,radius,axis=0)
    
    test_set = test_set.reshape((test_size,1,data_dsp_dim[0],data_dsp_dim[1]))
    label_set = label_set.reshape((test_size,1,label_dsp_dim[0],label_dsp_dim[1]))
    radius_set = radius_set.reshape((test_size,1,1,len(position)))

    if velocity_flag:
        test_set = (test_set - test_set.min())/(test_set.max() - test_set.min())
        label_set = (label_set - label_set.min())/(label_set.max() - label_set.min())
    return test_set, label_set, data_dsp_dim, label_dsp_dim, radius_set

# downsampling function by taking the middle value
def decimate(a,axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1,-1)]+1.0)/2.0-1).reshape(-1)
    downa = np.array(a)[:,:,idx[0].astype(int),idx[1].astype(int)]
    return downa
