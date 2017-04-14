# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from lib.vocabulary import *
from sym import caption_module
import mxnet as mx
import argparse
import logging
import os
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoches', default=20, type=int, help="epoches in training-stage", dest='epoches')
    parser.add_argument('--batch_size', default=50, type=int, help="batch_size in training-stage", dest='batch_size')
    parser.add_argument('--num_hidden', default=256, type=int, help="the number of hidden unit", dest='num_hidden')
    parser.add_argument('--lr', default=0.1, type=float, help="learning rate in training-stage", dest='lr')
    parser.add_argument('--num_embed', default=256, type=int, help="the number of embedding dimension", dest='num_embed')
    parser.add_argument('--num_lstm_layer', default=1, type=int, help="the number of hidden_unit", dest='num_lstm_layer')
    parser.add_argument('--gpus', default=None, type=str, help="wether run on gpu device", dest='gpu')
    parser.add_argument('--prefix', default='./checkpoint/train', type=str, help="prefix of save checkpoint", dest='prefix')
    parser.add_argument('--period', default=5, type=int, help="times to save checkpoint in training-stage", dest='period')
    return parser.parse_args()

args = parse_args()

processed_input_data_dir = 'processed_data'
checkpoints_prefix = args.prefix
if not os.path.exists(checkpoints_prefix):
    os.mkdir(checkpoints_prefix)

with open(processed_input_data_dir+'/training_prefixes.npy', 'rb') as f:
    training_prefixes = np.load(f)
with open(processed_input_data_dir+'/training_targets.npy', 'rb') as f:
    training_targets = np.load(f)
with open(processed_input_data_dir+'/training_indexes.npy', 'rb') as f:
    training_indexes = np.load(f)
with open(processed_input_data_dir+'/training_images.npy', 'rb') as f:
    training_images = np.load(f)

with open(processed_input_data_dir+'/validation_prefixes.npy', 'rb') as f:
    validation_prefixes = np.load(f)
with open(processed_input_data_dir+'/validation_targets.npy', 'rb') as f:
    validation_targets = np.load(f)
with open(processed_input_data_dir+'/validation_indexes.npy', 'rb') as f:
    validation_indexes = np.load(f)
with open(processed_input_data_dir+'/validation_images.npy', 'rb') as f:
    validation_images = np.load(f)

with open(processed_input_data_dir+'/vocabulary.txt', 'r') as f:
    tokens = f.read().strip().split('\n')
prefix_vocabulary = Vocabulary(tokens, pad_index=0, start_index=1, unknown_index=-1)
target_vocabulary = Vocabulary(tokens, end_index=0, unknown_index=-1)

input_vocab_size  = len(tokens) + 3 #pad, start, unknown
output_vocab_size = len(tokens) + 2 #end, unknown
sentence_size = len(training_prefixes[0])

print("---------------INFO-----------------------")
print('vocab_size:{}'.format(input_vocab_size))
print("sentence_length:{}".format(sentence_size))
print("-----------------------------------------")

num_lstm_layer = args.num_lstm_layer
num_hidden = args.num_hidden
num_embed = args.num_embed
batch_size = args.batch_size
caption_sym = caption_module(num_lstm_layer=num_lstm_layer, seq_len=sentence_size, vocab_size=input_vocab_size, 
                            num_hidden=num_hidden, num_embed=num_embed, batch_size=batch_size)
train_iter = mx.io.NDArrayIter(
        data={
            'image_feature': training_images,
            'word_data': training_prefixes
            },
        label={'softmax_label': training_targets},
        batch_size=batch_size,
        shuffle=True
    )

val_iter = mx.io.NDArrayIter(
        data={
            'image_feature': validation_images,
            'word_data': validation_prefixes
            },
        label={'softmax_label': validation_targets},
        batch_size=batch_size,
        shuffle=True
    )

ctx = mx.cpu() if not args.gpus else [mx.gpu(int(i)) for i in args.gpus.split(',')]
m = mx.mod.Module(symbol=caption_sym, data_names=('image_feature', 'word_data'), context=ctx)
m.fit(
    train_data=train_iter, 
    eval_data=val_iter,
    eval_metric=mx.metric.Perplexity(ignore_label=-1), 
    num_epoch=10, 
    batch_end_callback=mx.callback.Speedometer(batch_size=50, frequent=100),
    epoch_end_callback=mx.callback.do_checkpoint(checkpoints_prefix, period=10)
    )


