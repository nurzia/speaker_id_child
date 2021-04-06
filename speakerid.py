import argparse
import glob
import json
import os
import random
import pandas as pd

import operator
from itertools import islice

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import loading
import modelling
import numpy as np
from enrich import create_enriched_file




random.seed(23217)
"""
CUDA_VISIBLE_DEVICES="1" python3.6 speakerid.py --folder='data/' --num_epochs=30 --cutoff=740  
--enriched_file_train='enr_file_train.csv' --enriched_file_test='enr_file_test.csv' --report_file='acc_log.dat'
"""

def run_model(streamer, num_classes, spec_len, 
              lr, epochs, feat_dim):

    # definition and comp. of model, checkpoint, learning rate, optimizer, 
    model = modelling.build_model_(num_classes=num_classes,
                                   spec_len=spec_len, feat_dim=feat_dim)
    checkpoint = ModelCheckpoint('speaker_id.h5', monitor='val_loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                   patience=1, min_lr=0.00001,
                                   verbose=1, min_delta=0.001)
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0.0, amsgrad=False,
                           clipvalue=0.5, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['categorical_accuracy'])
    # fit model to data
    model.fit_generator(\
              streamer.scale_batches(stream='train', endless=True),
              steps_per_epoch=streamer.batch_lens['train'],
              epochs=epochs,
              validation_data=streamer.scale_batches(stream='dev', endless=True),
              validation_steps=streamer.batch_lens['dev'],
              callbacks=[checkpoint, reduce_lr])

    # model evaluation on test set
    return model

def create_reports(streamer, model):

    labels, instances = [], []
    for batch, batch_labels in \
        streamer.scale_batches(stream='test', endless=False):
        labels += list(batch_labels)
        instances += list(batch)

    instances = np.array(instances)
    labels = np.array(labels)

    predictions = model.predict(instances)

    uncat_labels = [np.argmax(label) for label in labels]
    uncat_predictions = [np.argmax(pred) for pred in predictions]

    occurrences = np.zeros(len(streamer.encoder.classes_))
    for label in labels:
        occurrences[np.argmax(label)] += 1

    # show classification report
    clas_rep = classification_report(uncat_labels, uncat_predictions)
    conf_mat = confusion_matrix(uncat_labels, uncat_predictions)
    acc_sco = accuracy_score(uncat_labels, uncat_predictions)

    return clas_rep, conf_mat, acc_sco, \
    	   occurrences, uncat_labels, predictions, \
    	   uncat_predictions, instances

def create_kids_list(meta, kids_num):

    """creates kids list based on the highest number of utterances in data-set"""

    chi_lengths = {i:len(meta.loc[meta['child']==i]) for i in set(meta['child'])}
    chi_lengths = {k: v for k, v in sorted(chi_lengths.items(), 
                   key=lambda item: item[1], reverse=True)}
    def take(n, iterable):
        """Return first n items of the iterable as a list"""
        return list(islice(iterable, n))
    chi_lengths = take(kids_num, chi_lengths.items())
    kids_list = [i[0] for i in chi_lengths]
    cutoff = chi_lengths[-1][1]
    return kids_list, cutoff


def create_args():

    parser = argparse.ArgumentParser(description='Trains a speaker id system')
    parser.add_argument('--folder', type=str,
                        default='data/',
                        help='Location of the annotated data folder')
    parser.add_argument('--seed', type=int, default=87487,
                        help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--cutoff', type=int, default=None,
                        help='cutoff')
    parser.add_argument('--pad', default='interval', type=str,
                        help='decides whether to mean-pad or max-pad')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--min_reduction', type=float, default=None,
                        help='minimum length to be included')
    parser.add_argument('--max_reduction', type=float, default=None,
                        help='maximum length to be included')
    parser.add_argument('--enriched_file_train', type=str,
                        default='enr_file_train.csv',
                        help='file with phonetic features of train set')
    parser.add_argument('--enriched_file_dev', type=str,
                        default='enr_file_dev.csv',
                        help='file with phonetic features of validation set')
    parser.add_argument('--enriched_file_test', type=str,
                        default='enr_file_test.csv',
                        help='file with phonetic features of test set')
    parser.add_argument('--report_file', type=str,
                        default='new_log.dat',
                        help='file with class. report')
    parser.add_argument('--threshold', type=int, default=None,
                        help='age threshold')
    parser.add_argument('--below_over', type=str, default=None,
                        help='decides whether below or over age threshold')
    parser.add_argument('--kids_list', type=bool, default=False,
                        help='whether or not the kids list exists')
    parser.add_argument('--kids_num', type=int, default=10,
                        help='num of kids to be included if kids list not provided')

    args = parser.parse_args()
    return args

def main():

    # create keyboard arguments
    args = create_args()

    # all metadata to draw the three subsets from (train, dev, test)
    meta = pd.read_csv(args.folder + 'metadata_cut.csv')

    if not args.kids_list:
    	kids_list, cutoff = create_kids_list(meta=meta, kids_num=args.kids_num)
    else:
    	    # list of children to train on
    	kids_list = ['att', 'max', 'oon', 'mad', 'her', 'vic',
                     'bra', 'wou', 'chl', 'lot']
    	cutoff = args.cutoff
# ----------------------------------------------------------
    # call streamer class, fit its scaler and compute number of batches
    streamer = loading.DataGenerator(meta=meta,
                                     batch_size=args.batch_size, 
                                     seed=args.seed, 
                                     pad=args.pad, 
                                     cutoff=cutoff, 
                                     min_reduction=args.min_reduction,
                                     max_reduction=args.max_reduction,
                                     kids_list=kids_list, 
                                     threshold=args.threshold,
                                     below_over=args.below_over, 
                                     field='child', 
                                     folder=args.folder)
    streamer.fit_scaler()
    streamer.get_num_batches()

    # extract parameters from streamer class
    num_classes = len(streamer.encoder.classes_)
    feat_dim = streamer.scaler.mean_.shape[0]
    spec_len = streamer.pad_length

    # train and evaluate model
    model = run_model(streamer=streamer, num_classes=num_classes, 
                      spec_len=spec_len, feat_dim=feat_dim, 
                      lr=args.lr, epochs=args.num_epochs)
    results = model.evaluate_generator(\
                    streamer.scale_batches(stream='test', endless=True),
                    steps=streamer.batch_lens['test'])
# ----------------------------------------------------------

    # # create input file for the function that computes the phonetic features
    # meta.loc[meta['filename'].isin(\
    #   [i[0] for i in streamer.streams['train']])].to_csv('infile_train.csv')
    # meta.loc[meta['filename'].isin(\
    #   [i[0] for i in streamer.streams['dev']])].to_csv('infile_dev.csv')

# ----------------------------------------------------------
    # create and write to file the classification report, confusion matrix and accuracy score
    clas_rep, conf_mat, acc_sco, occurrences, uncat_labels, \
    predictions, uncat_predictions, instances = \
    create_reports(streamer=streamer, model=model)
# ----------------------------------------------------------

    # kids_list = list(streamer.encoder.classes_)
    # meta_test = \
    # pd.DataFrame(columns=list(meta.columns)+kids_list+['pred'])
    # for index, (instance, pred) in enumerate(zip(instances, predictions)):
    #     temp = meta.loc[meta['filename']==streamer.streams['test'][index][0]]
    #     for index, kid in enumerate(kids_list):
    #         temp[kid] = pred[index]
    #     temp['pred'] = list(kids_list)[np.argmax(pred)]
    #     print(temp)
    #     meta_test = pd.concat([meta_test, temp])

    # meta_test.to_csv('infile_test.csv')
# ----------------------------------------------------------
    with open(args.report_file, 'w') as acc_log:
        acc_log.write(str(clas_rep) + '\n' + 
                      str(conf_mat) + '\n' + 
                      str(acc_sco) + '\n' + 
                      str(occurrences))
# ----------------------------------------------------------
    # # compute phonetic features
    # create_enriched_file(in_meta='infile_train.csv', 
    #                      out_meta=args.enriched_file_train)
    # create_enriched_file(in_meta='infile_dev.csv', 
    #                      out_meta=args.enriched_file_dev)
    # create_enriched_file(in_meta='infile_test.csv',
    #                      out_meta=args.enriched_file_test)

if __name__ == '__main__':
    main()

