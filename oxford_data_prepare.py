import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
import numpy as np
import pandas as pd


class DataPrepare(Dataset):
    def __init__(self, train):
        self.len = train.shape[0]
        x_set = train[:, 0:-1]
        x_set = x_set.reshape(x_set.shape[0], 660, 4)
        # x_set = x_set.reshape(x_set.shape[0], 2640)
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(train[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


file_name1 = '/Users/baitianyou/Desktop/CNN-ASTLSTM-main/Oxford_Battery_Degradation_Dataset_1.mat'
raw_data = loadmat(file_name1)
# print(type(raw_data))
# raw_data = raw_data.values()
cell1_stc = raw_data['Cell8']
# print(cell1_void)
# print(type(cell1_void[0][0]))
# cell1_stc = cell1_void[0][0][0][0][0][1][0][0][1]
# print(cell1_stc)
# cell1 = np.zeros([5, len(cell1_stc[0][0]), len(cell1_stc[0][0][0][0][0][1][0][0][0]), 1])
# print(cell1.shape)
t = []
v = []
q = []
T = []
C = []
print(len(cell1_stc[0][0][0][0][0][1][0][0][0]))
for i in range(len(cell1_stc[0][0])):
    t.append(cell1_stc[0][0][i][0][0][1][0][0][0])
    v.append(cell1_stc[0][0][i][0][0][1][0][0][1])
    q.append(cell1_stc[0][0][i][0][0][1][0][0][2])
    T.append(cell1_stc[0][0][i][0][0][1][0][0][3])
    C.append(-cell1_stc[0][0][i][0][0][1][0][0][2].min())
cell1 = [t, v, q, T, C]
cell1 = pd.DataFrame(data=cell1)
cell1.to_csv('cell8.csv')
# cell1[0] = cell1_dict[:]['C1dc']['t']
# print(cell1_dict['cyc0000'])
# print(cell1)
