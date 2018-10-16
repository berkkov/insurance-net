import pandas as pd
import torch.utils.data
import torch
import csv
from torchvision import transforms
import torch.nn.functional as F

'''
dataset_list = []
with open('train.csv') as csvFile:
    reader = csv.reader(csvFile)
    for i in reader:
        del i[2]
        dataset_list.append(i)


print(dataset_list[1])
del dataset_list[0]
for i in range(len(dataset_list)):
    print(i)
    dataset_list[i] = list(map(float, dataset_list[i]))
'''


class InsuranceDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        data_frame = pd.read_csv(path)
        response = data_frame.pop('Response')
        data_frame = data_frame.drop(['Id', 'Product_Info_2'], axis=1)
        data_frame = data_frame.fillna(0)

        self.data_tensor = torch.from_numpy(data_frame.values)
        self.response = torch.from_numpy(response.values)
        self.response = self.response - 1
        means = (torch.load('mean'))
        stds = (torch.load('std'))
        self.data_tensor = (self.data_tensor - means) / stds

    def __getitem__(self, index):
        x = self.data_tensor[index]
        y = self.response[index]
        return x.float().cuda(), y.float().cuda()

    def __len__(self):
        return self.data_tensor.size()[0]
