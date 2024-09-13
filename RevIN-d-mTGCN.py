# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:14:23 2024

@author: 16963
"""

import pandas as pd
import numpy as np
import time
#from numpy import array, array_equal
import torch
import os
import matplotlib.pyplot as plt
#import tensorflow as tf
import random 
import math
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from RevIN import RevIN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv,ChebConv
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch, DynamicGraphTemporalSignalBatch
from modern_tcn import ModernTCN
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# 由于GPU计算过程会带入随机性，需要将其固定以便在多次训练后产生相同效果
rnd_seed = 22

def set_seed(seed_num) -> None:
    random.seed(seed_num)
    np.random.seed(seed_num)
    #tf.random.set_seed(seed_num)
    #tf.experimental.numpy.random.seed(seed_num)
    #tf.random.set_seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)  #设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同
    torch.backends.cudnn.deterministic = True  
    #Torch 的随机种子为固定值的话，可以保证每次运行网络的时候相同输入的输出是固定的，
    #当使用gpu训练模型时，可能引入额外的随机源，使得结果不能准确再现
    torch.backends.cudnn.benchmark = False
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_num)
    print(f"Random seed set as {seed_num}")

#将监测站点各项数据读取为pd.df
LA_data_train = pd.read_csv('./Input/LA_Station_train4392-19-5_no0.csv').iloc[:,:]  #(110808, 5),len=5832×station_num=19=110808，feature_num=5
LA_data_test = pd.read_csv('./Input/LA_Station_test4392-19-5_no0.csv').iloc[:,:]  #(55632, 5),len=2928×station_num=19=55632，feature_num=5

#将pd.df转换为numpy数组
fin_data=LA_data_train.to_numpy()
fin_data_test=LA_data_test.to_numpy()

## 归一化
train_maxs = np.max(fin_data, axis=(0))  #得到5个特征在所有站点和所有时间序列上的最大值
train_mins = np.min(fin_data, axis=(0))
print(train_maxs, train_mins)

test_maxs = np.max(fin_data_test, axis=(0))  #得到5个特征在所有站点和所有时间序列上的最大值
test_mins = np.min(fin_data_test, axis=(0))
print(test_maxs, test_mins)

fin_data_norm = (fin_data - train_mins.reshape(1, -1))/(train_maxs.reshape(1, -1) - train_mins.reshape(1, -1))
fin_data_test_norm = (fin_data_test - test_mins.reshape(1, -1))/(test_maxs.reshape(1, -1) - test_mins.reshape(1, -1))
#print(fin_data_test)

##将原来二维数组转换为目标的三维（19，8760，5）,(19, 4392, 5)
fin_data_norm=fin_data_norm.reshape(-1, 19, 5)
fin_data_test_norm=fin_data_test_norm.reshape(-1, 19, 5)
#print(fin_data.shape)   #(4392, 19, 5)
#print(fin_data[0,:,:])

## shape再转化，fin_data为原数据，转换成(19站点,5特征,4392时间序列)的shape
data = fin_data_norm.transpose(
            (1, 2, 0)
        )
data = data.astype(np.float32)
#print(data.shape)  #(19, 5, 4392)
#print(data[0,:,:])

data_norm= torch.from_numpy(data)  #站点各项数据
#print(data_norm[18,:,4367:4391])  #利用第19个站点最后一个窗口验证时间窗口feature划分的正确性
#print(data_norm[18,0,24:4392])  #利用第19个站点验证时间窗口target划分的正确性

# 利用训练集的均值与标准差对测试集进行归一化
data_test = fin_data_test_norm.transpose(
            (1, 2, 0)
        )
data_test = data_test.astype(np.float32)
#print(data_test[0,:,:])

# 将标准化后的测试集数据传入torch
data_test_norm = torch.from_numpy(data_test)
#print(data_test_norm[0,:,:])
#print(data_test_norm.shape)  #[19, 5, 4392] 即19各站点，每个站点五个特征，共4392个时间序列元素

def split_dataset(data, n_sequence, n_pre, time_step):
    '''
    对数据进行处理，根据输入模型的历史数据长度和预测数据长度对数据进行切分
    :param data: 输入数据集
    :param n_sequence: 历史数据长度
    :param n_pre: 预测数据长度
    :return: 划分好的train data和target data
    '''
    train_X, train_Y = [], []    
    for i in range(data.shape[2] - n_sequence - time_step):
        a = data[:, :, i:(i + n_sequence)]
        train_X.append(a)
        b = data[:, 0, (i + n_sequence + time_step):(i + n_sequence + time_step + n_pre)]
        train_Y.append(b)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    print('T+n=', time_step+1)
 
    return train_X, train_Y  #[19,5,24], [19,1]

history_data = 24
predict_data = 1
time_step = 23
train_feature, train_target = split_dataset(data_norm, history_data, predict_data, time_step)

#将节点多维特征向量合并为一个向量
#train_feature = train_feature.reshape(train_feature.shape[0],train_feature.shape[1],-1)
test_feature, test_target = split_dataset(data_test_norm, history_data, predict_data, time_step)
'''
train_feature = list(train_feature)
train_target = list(train_target)
test_feature = list(test_feature)
test_target = list(test_target)
'''
#将节点多维特征向量合并为一个向量
#test_feature = test_feature.reshape(test_feature.shape[0],test_feature.shape[1],-1)
#print(train_feature.shape, train_target.shape)#(4368, 19, 5×24) (4368, 19, 1)
#print(test_feature.shape, test_target.shape)#(4368, 19, 5×24) (4368, 19, 1)

#print(train_feature[4367,18,:]) #利用第19个站点最后一个窗口验证时间窗口feature划分的正确性
#print(train_target[:,18,:])  #利用第19个站点验证时间窗口target划分的正确性

##动态邻接矩阵计算与构建模块

#导入邻接矩阵
#adj_mat_complete_test = pd.read_csv('./Input/DTW-geod_adj_test2306_2311_cut0.2.csv')
#adj_mat_complete_train = pd.read_csv('./Input/DTW-geod_adj_train2206_2211_cut0.2.csv')
#adj_mat_complete = pd.read_csv('./Input/DTWgeod integrated normed.csv')
#adj_mat_complete = np.array(adj_mat_dtw)
adj_geod = pd.read_csv('./Input/distance integrated normed.csv')
#adj_mat_geod = np.array(adj_mat_geod)
print(adj_geod)

# 以O3数据为基础计算每个时间窗口中19个站点间的DTW
O3_train = train_feature[:,:,0,:]   #(4368, 19, 24)
O3_test = test_feature[:,:,0,:]
#print(O3_train[0,:,:])

dis_adj = np.array(adj_geod)

def get_Dynadj(O3data):
    adj = pd.DataFrame()
    for k in range(0,O3data.shape[0]):
        window_T = O3data[k, :, :]
        nodes_num = window_T.shape[0]
        #print(window_T.shape)  #(19, 24)
        #存储DTW-distance数据
        adj_raw = pd.DataFrame(np.zeros(shape=(window_T.shape[0], window_T.shape[0])))  #全零df

        # 寻找DTW的均值，并将DTWd-distance填入adj_raw对应位置
        DTW = []
        for i in range(0,nodes_num):
            for j in range(0,nodes_num):
                if i == j:
                    pass
                else:
                    # 在此计算时间窗口中各个站点之间的Distance
                    distance, _= fastdtw(window_T[i].reshape(24,1), window_T[j].reshape(24,1), dist=euclidean)
                    adj_raw.at[i,j] = distance
                    DTW.append(distance)
                    #adj_T.at[i,j]=1/alignment.distance
        DTW = np.array(DTW)
        adj_raw = np.array(adj_raw)
        #print(adj_raw)
        
        p = 2
        c = 0.25
        adj_norm = (1 - p * adj_raw/( (np.sum(DTW) / 2) / nodes_num) )+c
        #print(adj_norm)

        #adj_norm[adj_norm == np.inf] = 0
        #print(adj_norm)

        adj_T = adj_norm * dis_adj
        #cutoff
        adj_T[adj_T >= 0.999] = 0.999
        adj_T[adj_T <= 0.13] = 0
        #adj_T = np.around(adj_T, decimals=4)
        adj_T = pd.DataFrame(adj_T)
    
        adj = pd.concat([adj, adj_T])
    return np.array(adj)

'''
adj_mat_complete.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
adj_mat_complete.index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
print(adj_mat_complete)
#print(adj_mat_complete.max()) #存在＞1的值

# 将邻接矩阵转换为numpy
adj = adj_mat_complete.to_numpy()  #adj_mat_complete存储了上述的邻接矩阵
#print(adj.max(),adj.min())
'''

# get the start DTW time
st = time.time()

if os.path.isfile('./Input/DTW-geod_adj_train.csv'):
    print("=> loaded DTW-geod-adj_train")
    adj_train = np.array(pd.read_csv('./Input/DTW-geod_adj_train.csv'))
else:
    print("=> no DTW-geod-adj_train, Compute training-DTW-geod-adj next······")
    adj_train = get_Dynadj(O3_train)
    print('train_adj.shape:',adj_train.shape)
    pd.DataFrame(adj_train).to_csv('./Input/DTW-geod_adj_train.csv',index=None)

if os.path.isfile('./Input/DTW-geod_adj_test.csv'):
    print("=> loaded DTW-geod-adj_test")
    adj_test = np.array(pd.read_csv('./Input/DTW-geod_adj_test.csv'))
else:
    print("=> no DTW-geod-adj_test, Compute test-DTW-geod-adj next······")
    adj_test = get_Dynadj(O3_test)
    print('test_adj.shape:',adj_test.shape)
    pd.DataFrame(adj_test).to_csv('./Input/DTW-geod_adj_test.csv',index=None)

# get the end DTW time
ed = time.time()

# get the DTW-computing time
elapsed_time = ed - st
print('fastDTW-computing time:', elapsed_time, 'seconds')

def split_adj(data):
    n_node = 19
    k=0
    adj_test_T = []
    for i in range(0, test_feature.shape[0]):
        
        a = data[ 0+k : n_node+k , : ]
        adj_test_T.append(a)
        k += n_node
    
    adj_Dyn = np.array(adj_test_T)
 
    return adj_Dyn

adj_train_T = split_adj(adj_train)
adj_test_T = split_adj(adj_test)
print('train_adj.shape:',adj_train_T.shape)  #4368,19,19
print('test_adj.shape:',adj_test_T.shape)  #4368,19,19

#将邻接矩阵和标准化后的监测数据传入torch
adj_train_T = torch.from_numpy(adj_train_T)
adj_test_T = torch.from_numpy(adj_test_T)  #邻接矩阵
#print(adj_test_T.shape)  #4368,19,19
#print(adj_train_T[2].shape)

#从邻接矩阵提取边索引与边权重（距离倒数）
#处理训练集adj
edge_indices_train=[]
values_train=[]
for i in range(0,adj_train_T.shape[0]):
    edge_indices_train_T, values_train_T = dense_to_sparse(adj_train_T[i])  #稠密矩阵转稀疏矩阵
    
    edge_indices_train_T = np.array(edge_indices_train_T, dtype=np.float64)
    edge_indices_train_T = edge_indices_train_T.tolist()
    
    values_train_T = np.array(values_train_T, dtype=np.float64)
    values_train_T = values_train_T.tolist()
    
    edge_indices_train.append(edge_indices_train_T)    
    values_train.append(values_train_T)

#edge_indices_train = np.array(edge_indices_train)
#values_train = np.array(values_train)
#print(edge_indices_train.shape) #4368,2,342 即有171条连接边，171=18+17+……+1
#print(values_train.shape)

#print(edge_indices_train[2])
#print(values_train[2])

#print(len(values))  #每两个站点对应一个距离，一共171×2个距离

## 设定边(2,342)与边权重(342,)
edges_train = edge_indices_train
#print(edges_train.shape)
edge_weights_train = values_train
#print(edge_weights)  #(4368, 342)

#处理训练集adj
edge_indices_test=[]
values_test=[]
for i in range(0,adj_test_T.shape[0]):
    edge_indices_test_T, values_test_T = dense_to_sparse(adj_test_T[i])  #稠密矩阵转稀疏矩阵
    
    edge_indices_test_T = np.array(edge_indices_test_T, dtype=np.float64)
    edge_indices_test_T = edge_indices_test_T.tolist()
    
    values_test_T = np.array(values_test_T, dtype=np.float64)
    values_test_T = values_test_T.tolist()
    
    edge_indices_test.append(edge_indices_test_T)
    values_test.append(values_test_T)
    
#edge_indices_test = np.array(edge_indices_test)
#values_test = np.array(values_test)
#print(edge_indices_test.shape) #4368,2,342 即有171条连接边，171=18+17+……+1
#(values_test.shape)  (4368, 342)

#print(edge_indices_test[3])
#print(values_test[3])

#print(len(values))  #每两个站点对应一个距离，一共171×2个距离

## 设定边(2,342)与边权重(342,)
edges_test = edge_indices_test
#print(edges_test.shape)
edge_weights_test = values_test
#print(type(edges_test))  ##<class 'numpy.ndarray'>
#print(type(edge_weights_test))

batch = []
for i in range(0,adj_train_T.shape[0]):
    batch.append(64)

train_dataset = DynamicGraphTemporalSignalBatch(edge_indices=edges_train, edge_weights=edge_weights_train, features=train_feature, targets=train_target, batches=batch)
test_dataset = DynamicGraphTemporalSignalBatch(edge_indices=edges_test, edge_weights=edge_weights_test, features=test_feature, targets=test_target, batches=batch)

set_seed(rnd_seed)#调用设置随机种子函数

class TCN(torch.nn.Module):
    def __init__(self, node_features, hid_size, output_size, revin = False):
        super(TCN, self).__init__()
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(node_features, subtract_last=True)#,eps=0.0001)
             
        self.tcn = ModernTCN(M = node_features,
                             L = 24,
                             D = hid_size,
                             P = 6,
                             S = 3)
        
        self.relu1 = nn.ReLU()
        
        self.gcn_1 = ChebConv(
            in_channels=hid_size * 8,
            out_channels=hid_size * 8,
            K = 3)
        
        self.relu2 = nn.ReLU()
        
        self.MLP = torch.nn.Sequential(
                   torch.nn.Linear(hid_size*8, hid_size*8 // 2),
                   torch.nn.ReLU(),
                   torch.nn.Linear(hid_size*8 // 2, output_size))
                   
        #self.lin = torch.nn.Linear(output_size, output_size)
                   
        #self.upto0 = nn.Softplus()

    def forward(self, x, edge_index, edge_weight):
        
        if self.revin:
            x = x.permute(0, 2, 1)  #RevIN的输入:[Batch_size,seq_len,num_features]
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        h = self.tcn(x) #[B, M, D*N]
        h = self.relu1(h)
        
        h_c = self.gcn_1(h[:,0,:], edge_index, edge_weight) #[B, D*N]
        h_c = self.relu2(h_c)
        #h_c = h_c.unsqueeze(1)  #[B, 1, D*N]
        #ho = torch.cat([h_c,h[:,1:5,:]],axis=1)
        
        #out = self.MLP(ho) #[B, M, 1]
        out = self.MLP(h_c) #[B, M, 1]
                
        if self.revin:
            out = out.permute(0, 2, 1)
            out = self.revin_layer(out, 'denorm')
            out = out.permute(0, 2, 1)

        #return out[:,0,:]
        return out

# GPU训练
device = torch.device('cuda') # cpu

node_features = 5
hid_size = 12
output_size = 1

model = TCN(node_features, hid_size, output_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
#loss_mse = torch.nn.MSELoss()
#cost_list = []

import time
# get the start time
st = time.time()

model.train()
print(model)

strat_epoch = 0
resume = True
# 加载之前保存的模型状态
if resume:
    if os.path.isfile('NoRevIN-d-mTGCN_T+24_checkpoint'):
        checkpoint = torch.load('NoRevIN-d-mTGCN_T+24_checkpoint')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded NoRevIN-d-mTGCN_T+24_checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no NoRevIN-d-mTGCN_T+24_checkpoint found")

# 进行训练
print("Running training...")
for epoch in range(4): 
    loss = 0
    step = 0
    for snapshot in train_dataset: #利用数据集的每一张图快照用作训练输入，共len(train_feature)=4368张图
        snapshot = snapshot.to(device) 
        # 根据离散图输入信号x(node_num,feature_num,seq_num)获得预测值
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # 均方误差,为最大最小归一化的数据
        loss = loss + torch.mean((y_hat-snapshot.y)**2) 
        step += 1
        
    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
    
    # 保存模型状态
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'NoRevIN-d-mTGCN_T+24_checkpoint')

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

model.eval()
testloss = 0
teststep = 0
    
# Store for analysis
predictions = []
labels = []

for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    #  Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    yhat_reverse = y_hat * (test_maxs[0]-test_mins[0])+test_mins[0]  #预测值
    snapshot.y_reverse = snapshot.y * (test_maxs[0]-test_mins[0])+test_mins[0]  #真实值
    #  Mean Absolute Error
    testloss = testloss + torch.mean(torch.abs(yhat_reverse-snapshot.y_reverse))
    # Store for analysis below
    labels.append(snapshot.y_reverse)
    predictions.append(yhat_reverse)
    teststep += 1
    
testloss = testloss / (teststep+1)
testloss = testloss.item()
print("Test MAE: {:.4f}".format(testloss))

# it is calculated for all nodes
ALLNode_pred = []
ALLNode_true = []
for item in predictions:
  for node in range(19):
    for hour in range(1):
      ALLNode_pred.append(item[node][hour].detach().cpu().numpy().item(0))

for item in labels:
  for node in range(19):
    for hour in range(1):
      ALLNode_true.append(item[node][hour].detach().cpu().numpy().item(0))

#predictions = torch.tensor([item.cpu().detach().numpy() for item in predictions])
#predictions = np.array(predictions)
#print(predictions.shape)
#pd.DataFrame(predictions).to_csv('predictions.csv')

ALLNode_pred_np = np.array(ALLNode_pred)
print(ALLNode_pred_np.shape)
#pd.DataFrame(ALLNode_pred_np).to_csv('test_yhat_noresh.csv')
ALLNode_pred_np_resh = ALLNode_pred_np.reshape(-1, 19, 1) #4368,19,1
ALLNode_true_np = np.array(ALLNode_true)
#pd.DataFrame(ALLNode_true_np).to_csv('test_label_noresh.csv')
ALLNode_true_np_resh = ALLNode_true_np.reshape(-1, 19, 1)

pd.DataFrame(ALLNode_true_np_resh.reshape(ALLNode_true_np_resh.shape[0],ALLNode_true_np_resh.shape[1])).to_csv('test_label.csv')
pd.DataFrame(ALLNode_pred_np_resh.reshape(ALLNode_pred_np_resh.shape[0],ALLNode_pred_np_resh.shape[1])).to_csv('test_yhat.csv')

print('所有站点真实值构型：',ALLNode_true_np_resh.shape)

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
#ALLNode_true_np_resh = ALLNode_true_np_resh.flatten()
#ALLNode_pred_np_resh = ALLNode_pred_np_resh.flatten()
##########evaluate##############
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(ALLNode_true_np, ALLNode_pred_np))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(ALLNode_true_np, ALLNode_pred_np)
# calculate R^2 决定系数（拟合优度）
test_r2 = r2_score(ALLNode_true_np, ALLNode_pred_np)
# calculate MAPE
mape = mean_absolute_percentage_error(ALLNode_pred_np, ALLNode_true_np)
# calculate sMAPE
sMAPE = smape(ALLNode_true_np, ALLNode_pred_np)
# calculate R 标签与预测值的相关系数
test_r = pearsonr(ALLNode_true_np.flatten(), ALLNode_pred_np.flatten())

print('RMSE: %.3f' % rmse)
print('MAE: %.3f' % mae)
print('R^{2}: %.4f' % test_r2)
print('MAPE: %.4f' % mape)
print('sMAPE: %.4f' % sMAPE)
print('R:' ,test_r[0])

