import numpy as np
import os
from tqdm import tqdm
import pandas as pd


class Data():
    def __init__(self, data_type='train', max_files=None):
        if data_type == 'train':
            num_of_files = 18662
            y_map = pd.read_csv(
                os.getcwd()+'/dataset/train_kaggle.csv', sep=',', usecols=["Label"])
        if data_type == 'test':
            num_of_files = 6051

        if (max_files != None):
            num_of_files = min(max_files, num_of_files)

        max_x = 0
        x_width = 0

        # Get size of array
        for i in range(num_of_files):
            x = np.load(os.getcwd()+'/dataset/' + data_type +
                        '/' + data_type+'/' + str(i) + '.npy')
            if i == 0:
                x_width = x.shape[1]
            max_x = max(max_x, x.shape[0])

        self.x = np.zeros((num_of_files, max_x, x_width))
        self.y = np.zeros((num_of_files, 1))

        with tqdm(total=num_of_files, bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for i in range(num_of_files):
                x = np.load(os.getcwd()+'/dataset/' + data_type +
                            '/' + data_type+'/' + str(i) + '.npy')
                self.x[i, 0:0 + x.shape[0]] = x
                if data_type == 'train':
                    self.y[i, 0] = y_map.values[i][0]
                pbar.update(1)
                pbar.set_description(
                    'Loading ' + data_type + ' data' + ' %g' % (i+1) + '/'+str(num_of_files))


if __name__ == '__main__':
    train_data = Data(data_type='train')
    test_data = Data(data_type='test')
    print(train_data.x.shape)
    print(train_data.y.shape)
    print(test_data.x.shape)
    print(test_data.y.shape)
