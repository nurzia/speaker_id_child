import errno
import json
import random
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split as split
import cv2

from keras.utils import to_categorical
from math import inf, isnan

# data augmentation
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    def __init__(self, meta, field,
                 batch_size=128, seed=123626, 
                 pad='interval', cutoff=1000, 
                 CI_exclude=False, min_reduction=None, 
                 max_reduction=None, kids_list=None, 
                 threshold=None, below_over=None, 
                 train_names=None, dev_names=None, 
                 test_names=None, limit_kids=None, 
                 folder=''):

        np.random.seed(seed)
        random.seed(seed)
        self.folder = folder
        self.batch_size = batch_size
        self.batch_lens = {}

        if field == 'status':
            self.streams = self.init_streams(meta=meta,
                                             train_names=train_names, 
                                             dev_names=dev_names, 
                                             test_names=test_names, 
                                             cutoff=cutoff, 
                                             min_reduction=min_reduction, 
                                             max_reduction=max_reduction, 
                                             threshold=threshold, 
                                             below_over=below_over,
                                             pad=pad,
                                             field=field,
                                             limit_kids=limit_kids)
        elif field =='child':
            self.streams = self.init_stream(meta_=meta,
                                            cutoff=cutoff, 
                                            CI_exclude=CI_exclude, 
                                            min_reduction=min_reduction, 
                                            max_reduction=max_reduction, 
                                            kids_list=kids_list, 
                                            threshold=threshold, 
                                            below_over=below_over,
                                            pad=pad)

        self.encoder = LabelEncoder()
        self.encoder.fit([l for _, l in self.streams['train']])
        print('-> working on classes:', self.encoder.classes_)

    def init_streams(self, meta, train_names, dev_names,
                     test_names, cutoff, min_reduction,
                     max_reduction, threshold, below_over,
                     pad, field, limit_kids):

        data={}
        for (subset, subset_names) in \
            zip(('train', 'dev', 'test'), (train_names, dev_names, test_names)):
            data[subset] = self.adjust_hearing(meta =\
                                self.cutoff_limit(meta =\
                                    self.limit_meta(meta_=meta, 
                                                    CI_exclude=False, 
                                                    min_reduction=min_reduction, 
                                                    max_reduction=max_reduction, 
                                                    kids_list=subset_names, 
                                                    threshold=threshold, 
                                                    below_over=below_over),
                                    cutoff = cutoff),
                                limit_kids=limit_kids)

        self.init_settings(data=data, pad=pad)
        data = {i: list(zip(data[i]['filename'], data[i][field])) for i in data}
        return data

    def init_stream(self, meta_, cutoff, CI_exclude,
                    min_reduction, max_reduction, kids_list, 
                    threshold, below_over, pad):
        meta = meta_
        meta = self.limit_meta(meta_=meta, 
                               CI_exclude=CI_exclude, 
                               min_reduction=min_reduction, 
                               max_reduction=max_reduction,
                               kids_list=kids_list, 
                               threshold=threshold, 
                               below_over=below_over)
        data = self.data_split(meta=meta, 
                               cutoff=cutoff,
                               pad=pad)
        return data

    def data_split(self, meta, cutoff, pad):

        if cutoff is not None:
            children = set(meta['child'])
            children_meta = pd.DataFrame()
            for child in children:
                child_list = meta.loc[meta['child'] == child]
                if len(child_list.index) < cutoff:
                    continue
                else:
                    child_list = child_list.sample(cutoff, random_state=123)
                children_meta = pd.concat([children_meta, child_list])
            children_meta = children_meta.sample(frac=1)
            train_meta, dev_meta, test_meta = self.metadata_split(df=children_meta)
        else:
            train_meta, dev_meta, test_meta = self.metadata_split(df=meta)

        data = {'train': train_meta, 'dev': dev_meta, 'test': test_meta}
        self.init_settings(data=data, pad=pad)
        data = {i: list(zip(data[i]['filename'], data[i]['child'])) for i in data}
        return data

    def metadata_split(self, df, train_size=.8, seed=8786):

        train, rest = split(df, stratify=df['child'],
                            train_size=train_size,
                            random_state=seed, shuffle=True)
        dev, test = split(rest, stratify=rest['child'],
                          train_size=.5,
                          random_state=seed, shuffle=True)
        return train, dev, test

    def limit_meta(self, meta_, CI_exclude, 
                   min_reduction, max_reduction,
                   kids_list, threshold, below_over):

        meta = meta_
        meta = meta.loc[meta['filename'].isin\
                     ([os.path.basename(i)[:-3] + 'wav' for \
                        i in glob.glob(self.folder+'spec/*')])]
        if CI_exclude == True:
            meta = meta.loc[meta['status'] == 'NH']
        elif CI_exclude == 2:
            meta = meta.loc[meta['status'] == 'CI']
        if min_reduction:
            meta = meta.loc[meta['length'] > min_reduction]
        if max_reduction:
            meta = meta.loc[meta['length'] < max_reduction]
        if kids_list:
            meta = meta.loc[meta['child'].isin(kids_list)]
        if threshold and below_over and (below_over == 'below'):
            meta = meta.loc[meta['age'] < threshold]
        elif threshold and below_over and (below_over == 'over'):
            meta = meta.loc[meta['age'] > threshold]
        return meta

    def init_settings(self, data, pad):

        self.mean_length = np.mean([data[i]['length'].mean() for i in data])
        self.mean_length = int((self.mean_length/1000 - 0.023) / 0.015) + 1
        self.max_length = np.max([data[i]['length'].max() for i in data])
        self.max_length = int((self.max_length/1000 - 0.023) / 0.015) + 1
        self.stddev = np.mean([data[i]['length'].std() for i in data])
        self.stddev = int((self.stddev/1000 - 0.023) / 0.015) + 1
        if pad == 'max':
            self.pad_length = self.max_length
        elif pad == 'mean':
            self.pad_length = self.mean_length
        elif pad == 'interval':
            self.pad_length = int(self.mean_length + 1 * self.stddev)
        else:
            try:
                self.pad_length = int(pad)
            except ValueError:
                print('invalid padding length')
                return False

    def cutoff_limit(self, meta, cutoff):

        children = set(meta['child'])
        children_meta = pd.DataFrame()
        for child in children:
            child_list = meta.loc[meta['child'] == child]
            if len(child_list.index) > cutoff:
                child_list = child_list.sample(cutoff, random_state=123)
                children_meta = pd.concat([children_meta, child_list])
        return children_meta

    def adjust_hearing(self, meta, limit_kids):

        with open('hearing.json') as f:
            hearing = json.load(f)
        nh_num, ci_num = 0, 0
        nh_kids, ci_kids = [], []
        for child in set(meta['child']):
            if hearing[child] == 0:
                nh_num += 1
                nh_kids.append(child)
            elif hearing[child] == 1:
                ci_num += 1
                ci_kids.append(child)

        extra_children = []
        difference = abs(nh_num-ci_num)
        if (nh_num > ci_num):
            extra_children = random.sample(nh_kids, k=difference)
        elif (ci_num > nh_num):
            extra_children = random.sample(ci_kids, k=difference)            
        meta_adj = meta.loc[~meta['child'].isin(extra_children)]
        if limit_kids:
            exk = random.sample(set(meta_adj.loc[meta_adj['status']=='NH']['child']),
                                 limit_kids) + \
                  random.sample(set(meta_adj.loc[meta_adj['status']=='CI']['child']),
                                 limit_kids)
            meta_adj = meta_adj.loc[~meta['child'].isin(exk)]
        return meta_adj

    def pre_process(self, x_batch, y_batch, image_datagen):
        for xt, yt in image_datagen.flow(x_batch,#.reshape(-1, x_batch.shape[1], x_batch.shape[2], 1), 
                                         y_batch, 
                                         batch_size = len(x_batch)):
            return xt, yt

    def scale_batches(self, stream, endless, rgb=False, augmentation=False):

        while True:
            X, Y = [], []
            random.shuffle(self.streams[stream])
            for idx, (fn, y) in enumerate(self.streams[stream]):
                try:
                    spec = np.load(self.folder + 'spec/' + fn[:-4] + '.npy')
                    X.append(spec)
                    Y.append(y)
                except OSError as e:
                    if e.errno == errno.ENOENT:
                        continue
                    else:
                        raise
                if len(Y) == self.batch_size or idx == len(self.streams[stream]) - 1:
                    X = [self.scaler.transform(x) for x in X]
                    X = np.array([self.pad(spec=x, length=self.pad_length) for x in X])
                    Y = self.encoder.transform(Y)
                    Y = to_categorical(Y, num_classes=len(self.encoder.classes_))
                    if rgb:
                        X = np.array([cv2.cvtColor(np.float32(x),cv2.COLOR_GRAY2RGB) for x in X])
                        # create augmented data for current batch and return them
                        if augmentation:
                            img_gen = ImageDataGenerator(shear_range=0.1, rotation_range=50, 
                                                         width_shift_range=0.2, height_shift_range=0.2, 
                                                         fill_mode='reflect', horizontal_flip = True, 
                                                         vertical_flip = False)
                            for xt, yt in img_gen.flow(X, Y, batch_size=len(X)):
                                yield(xt, yt)
                        else:
                            yield(X, Y)
                    else:
                        yield(X, Y)
                    X, Y = [], []
            if not endless:
                break

    def pad(self, spec, length):

        if spec.shape[0] < length:
            padded = np.full((length, spec.shape[1]),
                             fill_value=0, dtype=np.float64)
            padded[:spec.shape[0], : spec.shape[1]] += spec
            return padded
        else:
            return spec[:length, :]

    def get_batches(self, stream):

        X, Y = [], []
        random.shuffle(self.streams[stream])
        for idx, (fn, y) in enumerate(self.streams[stream]):
            try:
                spec = np.load(self.folder + 'spec/' + fn[:-4] + '.npy')
                X.append(spec)
                Y.append(y)
            except OSError as e:
                print('error in batch: ', self.folder + 'spec/' + fn + '.npy')
                if e.errno == errno.ENOENT:
                    continue
                else:
                    raise
            if len(Y) == self.batch_size or \
               idx == len(self.streams[stream]) - 1:
                yield (X, Y)
                X, Y = [], []

    def fit_scaler(self, stream='train'):
        self.scaler = StandardScaler()
        for X, _ in self.get_batches(stream):
            self.scaler.partial_fit(np.vstack(X))

    def get_num_batches(self):
        for stream in ('train', 'dev', 'test'):
            if not (stream in self.batch_lens):
                self.batch_lens[stream] = 0
                for batch, _ in self.get_batches(stream):
                    self.batch_lens[stream] += 1
