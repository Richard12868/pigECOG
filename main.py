from train_model import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from dataprocess import DataProcess
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import torch as t
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


import dill
from sklearn.ensemble import RandomForestRegressor
from sklearn import  metrics
import pickle
import torch.nn.functional as F

def traincFun(trainDataset,epochs):


    model=LSTM(input_size=63, hidden_size=63, out_features=1, num_layers=1)
    # model = logLSTM(input_width, hidden_size=hiddensize,out_features=15, num_layers=1)
    model = nn.DataParallel(model, device_ids=[0])

    loss_function = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.00005)
    model.train()
    once_train_loss_list=[]
    once_test_loss_list = []
    train_loss_list=[]
    test_loss_list = []
    for epoch in range(epochs):
        model.train()
        for step, (data, label) in enumerate(trainDataset):
            opt.zero_grad()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        # for name, parms in model.named_parameters():
                        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
                        #     )
                        # train_x = data.to(device)ize len序列数 每个元素的特征}


            train_x = data.to(device)
            train_y = label.to(device)
            print('x', train_x.shape)
            print('lable',train_y.shape)
            out = model(train_x)

            # print(train_y)
            print('out1',out.shape)
            # print(out)
            # out=np.squeeze(out)
            # print('out2',out.shape)

            # loss = loss_function(out, train_y)

# lstm
#             out = model.forward(train_x)
#             print('lable',train_y.shape)
#             print('out1',out.shape)
#             out=np.squeeze(out)
#             print('out2',out.shape)
            print('train_y',train_y.shape)
            loss = loss_function(out, train_y)

            loss.backward()
            opt.step()
            print('\rEpoch: ', epoch,'| Step: ',step, '| trainloss: ', loss.item(),end='')
            once_train_loss_list.append(loss.item())

    plt.plot(once_train_loss_list[:], color='red', label='loss')
    # plt.title(hiddensize)
    plt.show()
        # for step, (data, label) in enumerate(trestDataset):
        #     output = model(data)
        #     test_loss = loss_function(output, label)
        #     once_test_loss_list.append(test_loss.item())
        #
        # train_loss_list.append(np.mean(once_train_loss_list))
        # test_loss_list.append(np.mean(once_test_loss_list))
        # print('Epoch: ', epoch, '| avgtrainloss: ', train_loss_list[-1],
        #       '| avgtestloss: ', test_loss_list[-1])
    t.save(model, 'model_lstm.pth')
    return model

def testFunc(testDataset, model_path):
    model=torch.load(model_path)
    model.eval()  # 将模型设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out_list=torch.tensor([])
    testy_list=torch.tensor([])
    for step, (data, label) in enumerate(testDataset):

        test_x = data.to(device)
        test_y = label

        out = model(test_x).cpu()
        # print('out',out.shape)
        # print('test_y',test_y.shape)
        out=out*120
        # out[:, 1] = out[:, 1] * 120
        # out[:,2] = out[:,2] * 180
        # out[:,3] = out[:,3] * 180

        test_y = test_y * 120
        # test_y[:, 1] = test_y[:, 1] * 120
        # test_y[:, 2] = test_y[:, 2] * 180
        # test_y[:, 3] = test_y[:, 3] * 180

        testy_list = torch.cat((testy_list, test_y), dim=0)
        out_list= torch.cat((out_list, out), dim=0)

        print('out', type(out))
        print('test_y', type(test_y))

    print('out_list', out_list)
    print('testy_list', testy_list)
    # r2 = r2_score(out_list[440:480].detach().numpy(), testy_list[440:480].detach().numpy())
    plt.figure()
    j=790

    # def moving_average(data, window_size):
    #     # 确保窗口大小不超过数据长度
    #     window_size = min(window_size, len(data))
    #     # 计算滑动平均
    #     moving_avg = np.convolve(data.ravel(), np.ones(window_size) / window_size, mode='valid')
    #     return moving_avg

    # 原始数据
    # data = out_list[j:j + 28].detach().numpy()
    #
    # # 时间轴
    # t = np.arange(len(data)) * 0.28
    #
    # # 绘制原始曲线
    # plt.plot(t, data.ravel(), color='red', label='Original')

    # 滑动平均窗口大小（可以根据需要调整）
    # window_size = 2
    #
    # # 计算滑动平均
    # smoothed_data = moving_average(data, window_size)
    #
    # # 绘制平滑后的曲线
    # plt.plot(t[window_size - 1:], smoothed_data, color='blue', label=f'Smoothed (window size={window_size})')
    # plt.plot(t[window_size - 1:],testy_list[j:j + 26].detach().numpy(), color='red')
    # plt.legend()
    # plt.show()
    j=630
    t = np.arange(len(data))*0.28
    plt.figure()
    plt.plot(t[j:j + 28]-t[j],out_list[j:j + 28].detach().numpy(), color='red')
    plt.plot(t[j:j + 28]-t[j],testy_list[j:j + 28].detach().numpy(), color='blue')
    # plt.title(f'{j}')
    plt.xlabel('time/s')
    plt.ylabel('deg')
    plt.show()
    # for j in range(0,len(out_list),30):
    #     plt.figure()
    #     plt.plot(out_list[j:j+30].detach().numpy(), color='red')
    #     plt.plot(testy_list[j:j+30].detach().numpy(), color='blue')
    #     plt.title(f'{j}')
    #     plt.show()
    # print(f'-R² value: {r2}')
    # for i in range(out_list.shape[1]):
    #     # print(out_list[:10,i].detach().numpy(), testy_list[:10,i].detach().numpy())


    # mse_error = F.mse_loss(out, test_y )
        #
        # print("MSE Error1:", mse_error.item())

        # print('out',out)

#
#

if __name__ == '__main__':
    flag=False
    if flag==True:
        trainpath=r'./res_data/train_data/'
        inputsize=int(280/1000*2000)
        window_in=int(280/1000*2000)
        dataLoaderTrain=DataProcess(trainpath,inputsize,window_in).read_file()
        model=traincFun(dataLoaderTrain,300)
    else:
        test_path=r'./res_data/vali_data/'
        inputsize=int(280/1000*2000)
        window_in =int(280/1000*2000)
        dataLoaderTest=DataProcess(test_path,inputsize,window_in).read_file()
        model_path='model_lstm.pth'
        model=testFunc(dataLoaderTest, model_path)
