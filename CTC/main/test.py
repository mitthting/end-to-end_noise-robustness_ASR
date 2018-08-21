#-*- coding:utf-8 -*-
#!/usr/bin/python
''' Automatic Speech Recognition for LibriSpeech corpus

Original author(s):
zzw922cn

Modified from https://github.com/zzw922cn/Automatic_Speech_Recognition

'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import time
import datetime
import os
from six.moves import cPickle
from functools import wraps
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import getAttrs
from utils.utils import output_to_sequence
from utils.utils import list_dirs
from utils.utils import logging
from utils.utils import count_params
from utils.utils import target2phoneme
from utils.utils import get_edit_distance
from utils.taskUtils import get_num_classes
from utils.taskUtils import check_path_exists
from utils.taskUtils import dotdict
from utils.functionDictUtils import model_functions_dict
from utils.functionDictUtils import activation_functions_dict
from utils.functionDictUtils import optimizer_functions_dict

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

flags.DEFINE_string('task', 'end-to-end', 'set task name of this program')
flags.DEFINE_string('test_dataset', 'clean1', 'set the test dataset')
flags.DEFINE_string('mode', 'test', 'set whether to train, dev or test')
flags.DEFINE_boolean('keep', True, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('level', 'cha', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 3, 'set the layers for rnn')
flags.DEFINE_string('activation', 'tanh', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')
flags.DEFINE_integer('batch_size', 40, 'set the batch size')
flags.DEFINE_integer('num_hidden', 133, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 13, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 15, 'set the number of epochs')
flags.DEFINE_float('lr', 0.001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.5, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1.0, 'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_string('datadir', '/home/ding/Forgithub/Output_DAE_AFE_lossweight/testA', 'set the data root directory')
flags.DEFINE_string('labeldir', '/home/ding/Forgithub/CTC/label/testA/', 'set the label root directory')
flags.DEFINE_string('logdir', '/home/ding/Forgithub/CTC/log', 'set the log directory')

FLAGS = flags.FLAGS

labeldir = FLAGS.labeldir 

task = FLAGS.task

test_dataset = FLAGS.test_dataset

level = FLAGS.level
model_fn = model_functions_dict[FLAGS.model]

rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

activation_fn = activation_functions_dict[FLAGS.activation]
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature

num_classes = FLAGS.num_classes+1
num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
grad_clip = FLAGS.grad_clip
datadir = FLAGS.datadir

logdir = FLAGS.logdir
savedir = os.path.join(logdir, 'save')
resultdir = os.path.join(logdir, 'result')
loggingdir = os.path.join(logdir, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

mode = FLAGS.mode
keep = FLAGS.keep
keep_prob = 1-FLAGS.dropout_prob

print('%s mode...'%str(mode))
if mode == 'test' :
    batch_size = 143      #1001 = 7x11x13
    num_epochs = 1


def get_data(datadir, level, test_dataset, mode) :

    if mode == 'test':

        test_feature_dirs = [os.path.join(datadir, test_dataset)]

        test_label_dirs = [os.path.join(labeldir, test_dataset)]

    return test_feature_dirs, test_label_dirs

logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(), 
'%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))

#for testA and testB
SNRdir = ['clean1', 'N1_SNR0', 'N1_SNR5', 'N1_SNR10', 'N1_SNR15', 'N1_SNR20', 'N1_SNR-5',
	   'clean2', 'N2_SNR0', 'N2_SNR5', 'N2_SNR10', 'N2_SNR15', 'N2_SNR20', 'N2_SNR-5',
	   'clean3', 'N3_SNR0', 'N3_SNR5', 'N3_SNR10', 'N3_SNR15', 'N3_SNR20', 'N3_SNR-5',
	   'clean4', 'N4_SNR0', 'N4_SNR5', 'N4_SNR10', 'N4_SNR15', 'N4_SNR20', 'N4_SNR-5',]
'''
#for testC
SNRdir = ['clean1', 'N1_SNR0', 'N1_SNR5', 'N1_SNR10', 'N1_SNR15', 'N1_SNR20', 'N1_SNR-5',
	   'clean2', 'N2_SNR0', 'N2_SNR5', 'N2_SNR10', 'N2_SNR15', 'N2_SNR20', 'N2_SNR-5',]
'''

class Runner(object):

	def _default_configs(self):
	    return {'level': level,
	                 'rnncell': rnncell,
	                 'batch_size': batch_size,
	                 'num_hidden': num_hidden,
	                 'num_feature': num_feature,
	                 'num_class': num_classes,
	                 'num_layer': num_layer,
	                 'activation': activation_fn,
	                 'optimizer': optimizer_fn,
	                 'learning_rate': lr,
	                 'keep_prob': keep_prob,
	                 'grad_clip': grad_clip,
	               }

	def load_data(self, feature_dir, label_dir, mode, level) :
	    return load_batched_data(feature_dir, label_dir, batch_size, mode, level)
	    
	def run(self):
	    # load data
	    for test in range(len(SNRdir)) :

	        args_dict = self._default_configs()
	        args = dotdict(args_dict)
	    
	        feature_dirs, label_dirs = get_data(datadir, level, SNRdir[test], mode)

	        filenamestr = os.listdir(feature_dirs[0])

	        FL_pair = list(zip(feature_dirs, label_dirs))

	        feature_dirs, label_dirs = zip(*FL_pair)

	        for feature_dir, label_dir in zip(feature_dirs, label_dirs):

	            batchedData, maxTimeSteps, totalN = self.load_data(feature_dir, label_dir, mode, level)

	            model = model_fn(args, maxTimeSteps)
	    
	            with tf.Session(graph=model.graph) as sess :
	                # restore from stored model
	                if keep == True :
	                    ckpt = tf.train.get_checkpoint_state(savedir)
	                    if ckpt and ckpt.model_checkpoint_path:
	                        model.saver.restore(sess, ckpt.model_checkpoint_path)
	                        print('Model restored from:' + savedir)

	                for epoch in range(num_epochs):

	                    fnindex = 0  #used for write file name while testing

	                    batchErrors = np.zeros(len(batchedData))
	                
	                    batchRandIxs = np.arange(len(batchedData))

	                    for batch, batchOrigI in enumerate(batchRandIxs):
	                        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
	                        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse

	                        feedDict = {model.inputX : batchInputs, 
	                                           model.targetIxs : batchTargetIxs,
	                                           model.targetVals : batchTargetVals, 
	                                           model.targetShape : batchTargetShape,
	                                           model.seqLengths : batchSeqLengths}

	                        if level == 'cha' :

	                            if mode == 'test' :

	                                pre, er = sess.run([model.predictions, model.errorRate], feed_dict=feedDict)

	                                batchErrors[batch] = er

	                                digit = output_to_sequence(pre, type = level)

	                                with open(os.path.join(resultdir, SNRdir[test] + '_results.MLF'), 'a') as result :

	                                    if batch < 1 :

	                                        result.write('#!MLF!#' + '\n')

	                                        for i in range(batch_size) :

	                                            head = '"*/'

	                                            tail = '.rec"'

	                                            a = filenamestr[fnindex].split('.')

	                                            fnindex = fnindex + 1

	                                            labelstr = head + a[0] + tail

	                                            result.write(labelstr + '\n')

	                                            decode= digit[i]

	                                            for j in range(len(decode)) :
	                                                result.write(decode[j] + '\n')

	                                            result.write('.' + '\n')

	                                print('\n{} mode, total:{},batch:{}/{},mean test CER={:.4f}\n'.format(level, totalN, batch+1, len(batchRandIxs), er/batch_size))


if __name__ == '__main__':
    runner = Runner()
    runner.run()
