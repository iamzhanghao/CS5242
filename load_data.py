import numpy as np
import os
from tqdm import tqdm
import pandas as pd


class Data():
    def __init__(self, type='train', max_files=None):
        if type == 'train':
            num_of_files = 18662
            y_map = pd.read_csv(
                os.getcwd()+'/dataset/train_kaggle.csv', sep=',', usecols=["Label"])
        if type == 'test':
            num_of_files = 6051

        if (max_files != None):
            num_of_files = min(max_files, num_of_files)
        self.x = np.array([])
        self.y = np.array([])
        with tqdm(total=num_of_files, desc="Loading "+type+' data', bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for i in range(num_of_files):
                x = np.load(os.getcwd()+'/dataset/' + type +
                            '/' + type+'/' + str(i) + '.npy')

                if i == 0:
                    self.x = x
                    if type == 'train':
                        self.y = np.full((x.shape[0], 1), y_map.values[0][0])

                else:
                    self.x = np.append(self.x, x, axis=0)
                    if type == 'train':
                        self.y = np.append(self.y, np.full(
                            (x.shape[0], 1), y_map.values[i][0]), axis=0)
                pbar.update(1)
                pbar.set_description(
                    'Loading '+type + ' data' + ' %g' % (i+1) + '/'+str(num_of_files))


data = Data(type='train', max_files=50)
print(data.x.shape)
print(data.y.shape)
