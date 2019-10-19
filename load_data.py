import numpy as np
import os
from tqdm import tqdm


class Data():
    def __init__(self, name='train', max_files=None):
        if name == 'train':
            num_of_files = 18662
        if name == 'test':
            num_of_files = 6051
        if (max_files != None):
            num_of_files = min(max_files, num_of_files)
        self.x = np.array([])
        self.y = np.array([])
        with tqdm(total=num_of_files, desc="Loading "+name+' data', bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for i in range(num_of_files):
                x = np.load(os.getcwd()+'/dataset/' + name +
                            '/' + name+'/' + str(i) + '.npy')
                if i == 0:
                    self.x = x

                else:
                    self.x = np.append(self.x, x, axis=0)
                pbar.update(1)
                pbar.set_description(
                    'Loading '+name + 'data' + ' %g' % i + '/'+str(num_of_files))


data = Data(name='train', max_files=100)
print(data.x.shape)
print(data.y.shape)
