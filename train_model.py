import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_features,num_layers):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  #  输入数据的特征维数，通常就是embedding_dim(词向量的维度)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        # 针对lstm网络参数初始化
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

        # self.norm = nn.LayerNorm(30)
        self.linear_layer = nn.Linear(in_features=input_size,
                                      out_features=out_features,
                                      bias=True)
        # self.apply(self.weight_init)

    def forward(self, x):

        out1, (h_n, h_c) = self.lstm(x)
        a, b, c = h_n.shape
        lstm_h=h_n.squeeze(0)

        out2 = self.linear_layer(lstm_h)
        return out2