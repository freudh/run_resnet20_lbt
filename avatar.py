#-*- coding: UTF-8 -*-
import os
import scipy.misc
import numpy as np
from glob import glob

class Avatar:

    def __init__(self):
        self.data = 'data.txt'
        self.lable = 'lable.txt' 


    def batch_data(self):
        data = open(self.data, 'r')
        imgs = []
        lines = data.readlines()
        for line in lines:
            for db in line.split():
                imgs.append(db)
        batches = np.zeros([49152])
        batches[::] = imgs
        return np.reshape(batches[::], [16, 3*32*32])

    def batch_lable(self):
        lable = open(self.lable, 'r')
        imgs = []
        lines = lable.readlines()
        for line in lines:
            for db in line.split():
                imgs.append(int(db))
        batches = np.zeros([160])
        batches[::] = imgs
        return np.reshape(batches[::], [16,10])

if __name__ == '__main__':
    avatar = Avatar()
    batch = avatar.batch_data()
    print(batch[0])
