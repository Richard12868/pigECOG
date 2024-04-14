import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from scipy import stats, signal
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
class DataProcess():
    def __init__(self, path,inputsize,window_in):
        self.path = path
        self.input_size=inputsize
        self.window_in=window_in
    def read_file(self):


        # 读取文件形成列表
        train_data = []
        train_label = []
        self.df_list = []
        scaler = StandardScaler()
        for file in os.listdir(self.path):
            print(file)
            df = pd.read_csv(self.path+ file)
            for m in range(4,df.shape[1]):
                df.iloc[:,m] = self.emg_filter_lowpass(df.iloc[:,m],200)


            for i in range(self.input_size, len(df), self.window_in):
                # print(i)
                label =df.iloc[i,1]
                data = df.iloc[i - self.input_size:i,4:]

                label=label/120
                # label.iloc[1] = label.iloc[ 1] / 120
                # label.iloc[2]=label.iloc[2]/180
                # label.iloc[3] = label.iloc[ 3] / 180
                # print('traindata',(data).shape)
                # print('labeldata', (label).shape)
                # print('traindata',data)
                # print('labeldata', label)

                train_data.append(np.array(data,dtype=np.float32))
                train_label.append(np.array(label,dtype=np.float32))

        print(len(train_data))
        print(len(train_label))

        tensor_data = Data.TensorDataset(
            t.from_numpy(np.array(train_data )),
            t.from_numpy(np.array(train_label)))
        # 把 dataset 放入 DataLoader
        dataLoaderTrain = Data.DataLoader(
            dataset=tensor_data,  # 数据，封装进Data.TensorDataset()类的数据
            batch_size=2048,  # 每块的大小
            shuffle=False,  # 要不要打乱数据 (打乱比较好)
            num_workers=2,  # 多进程（multiprocess）来读数据
        )
        return dataLoaderTrain

    def emg_filter_lowpass(self,x,highcut, order = 4, sRate = 2000.):
        """ Forward-backward band-pass filtering (IIR butterworth filter) """
        nyq = 0.5 * sRate
        high =highcut/nyq

        b, a = signal.butter(order,high, 'lowpass')
        return signal.filtfilt(b,a,x)
# if __name__ == '__main__':
#     path=r'./res_data/train_data/'
#     inputsize=int(200/1000*2000)
#     dataLoaderTrain=DataProcess(path,inputsize).read_file()
    # print(dataLoaderTrain)


