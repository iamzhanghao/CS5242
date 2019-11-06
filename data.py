import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import csv


class Data():
    def __init__(self, data_type='train', max_files=None, normalize=True):
        self.normalize_factors = self.__get_persisted_normalize_factors()
        self.api_size_list = [8, 4, 16, 16, 8, 12, 16, 12] + [1] * 10
        
        if data_type == 'train':
            num_of_files = 18662
            y_map = pd.read_csv(os.getcwd()+'/dataset/train_kaggle.csv', sep=',', usecols=["Label"])
        if data_type == 'test':
            num_of_files = 6051
            if not self.normalize_factors:
                raise AttributeError("No normalize factor found to normalize test data. Please load training data at least once to generate normalize factors.")

        if (max_files != None):
            num_of_files = min(max_files, num_of_files)

        max_x = 1000
        x_width = 102

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

        # Normalize input features
        if (normalize):
            self.__normalize_x()
        
    def __get_persisted_normalize_factors(self):
        try:
            normalize_factors = []
            with open(os.getcwd()+'/dataset/normalize_factor.csv', 'rt') as normalize_factor_file:
                csv_reader = csv.reader(normalize_factor_file)
                for row in csv_reader:
                    normalize_factors.append(float(row[0]))
                return normalize_factors
        except FileNotFoundError:
            return []
        
    def __generate_normalize_factors(self):
        current_pos = 0
        for api_size in self.api_size_list:
            normalize_factor = self.__get_normalize_factor(self.x[:, :, current_pos:current_pos+api_size])
            self.normalize_factors.append(normalize_factor)
            current_pos += api_size
            
    def __get_normalize_factor(self, feature):
        abs_max_value = np.max(abs(feature))
        if not np.isscalar(abs_max_value):
            abs_max_value = np.max(abs_max_value)
        return abs_max_value
            
    def __persist_normalize_factors(self):
        with open(os.getcwd()+'/dataset/normalize_factor.csv', 'wt', newline='', encoding='utf-8') as normalize_factor_file:
            csv_writer = csv.writer(normalize_factor_file)
            csv_writer.writerows(map(lambda x:[x], self.normalize_factors))
    
    def __normalize_x(self):       
        if not self.normalize_factors:
            print("No existing normalize factors found. Generating new ones and persisting.")
            self.__generate_normalize_factors()
            self.__persist_normalize_factors()
        
        print("Normalizing x data using normalize factors: ", self.normalize_factors)
        current_pos = 0
        for index, api_size in enumerate(self.api_size_list):
            self.x[:, :, current_pos:current_pos+api_size] = self.x[:, :, current_pos:current_pos+api_size] / self.normalize_factors[index]
            current_pos += api_size
        

if __name__ == '__main__':
    train_data = Data(data_type='train', max_files=100)
    # test_data = Data(data_type='test')
    # print(train_data.x.shape)
    # print(train_data.y.shape)
    # print(test_data.x.shape)
    # print(test_data.y.shape)
    # train_data = Data(data_type='train',max_files=1000)
#     train_data = Data(data_type='train',max_files=10000)
    # test_data = Data(data_type='test')

#     frames =[]
#     with tqdm(total=len(train_data.x), bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
#         for i in range(len(train_data.x)):
#             for row in train_data.x[i]:
#                 if row[0:4].tolist() not in frames :
#                     frames.append(row[0:4].tolist())
#             pbar.update(1)
#             pbar.set_description(
#                     'Loading data' + ' %g' % (i+1) + '/'+str(len(train_data.x)))
#     print(frames)