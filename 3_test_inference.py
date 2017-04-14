# -*- conding=utf-8 -*-

"""
test module
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from lib.vocabulary import *
from sym import caption_module
import mxnet as mx
import argparse
import logging
import os
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--eval", default=True, type=bool, help='wether to evaluation')
parser.add_argument("--infer", default=True, type=bool, help='wether to inference')
parser.add_argument("--gen_prob", default=True, type=bool, help='wether to igenerate probablity')
parser.add_argument("--batch_size", default=1, type=int, help='batch size of test batch when it is in test stage')
args = parser.parse_args()

logging.info('load test data')
processed_input_data_dir = 'processed_data'
with open(processed_input_data_dir+'/test_grouped_prefixes.npy', 'rb') as f:
    test_grouped_prefixes = np.load(f)
with open(processed_input_data_dir+'/test_grouped_targets.npy', 'rb') as f:
    test_grouped_targets = np.load(f)
with open(processed_input_data_dir+'/test_grouped_images.npy', 'rb') as f:
    test_grouped_images = np.load(f)

with open(processed_input_data_dir+'/vocabulary.txt', 'r') as f:
    tokens = f.read().strip().split('\n')
prefix_vocabulary = Vocabulary(tokens, pad_index=0, start_index=1, unknown_index=-1)
target_vocabulary = Vocabulary(tokens, end_index=0, unknown_index=-1)
input_vocab_size  = len(tokens) + 3 #pad, start, unknown
output_vocab_size = len(tokens) + 2 #end, unknown


def load_model(prefix, epoch, ctx, batch_size):

    mod = mx.mod.Module.load(prefix=prefix, epoch=epoch, context=ctx, data_names=('image_feature', 'word_data'))
    data_shapes = [('image_feature', (batch_size, 4096)), ('word_data', (batch_size, 38))]
    label_shapes = [('softmax_label', (batch_size,))]
    mod.bind(data_shapes=data_shapes, label_shapes=label_shapes)
    return mod

def Batch(image, word, target):
    data_iter = mx.io.NDArrayIter(
        data={
            'image_feature': image,
            'word_data': word
            },
        label={'softmax_label': target},
        batch_size=1
    )
    return data_iter

m = load_model(prefix='checkpoint/checkpoint', epoch=10, ctx=mx.cpu(), batch_size=args.batch_size)

if args.eval:
    eval_dir = 'outputs'
    logging.info('score test...')
    score = []
    prob = []
    for image, prefix_group, target_group in zip(test_grouped_images, test_grouped_prefixes, test_grouped_targets):
        image_repeats = image.repeat(len(prefix_group), axis=0)
        data_batch = Batch(image=image_repeats, word=prefix_group, target=target_group)
        score.append(m.score(data_batch, mx.metric.Perplexity(-1))[0][1])
        if args.gen_prob:
            prob.append(m.predict(data_batch, num_batch=1))
        del data_batch
        if len(score) % 100 == 0:
            logging.info('currtent samples {}, mean-score {}'.format(len(score), np.mean(score)))
    prob_np = map(lambda x: x.asnumpy(), prob)

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    with open(eval_dir + 'prob.txt', 'w') as f:
        for row in prob_np:
            print(*[ str(p) for p in row ], sep='\t', file=f)
    
if args.infer:
    pass


