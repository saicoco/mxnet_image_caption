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
from lib.langmod_tools import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--eval", default=False, type=bool, help='wether to evaluation')
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


def load_model(prefix, epoch, ctx, batch_size, evalua=True):

    mod = mx.mod.Module.load(prefix=prefix, epoch=epoch, context=ctx, data_names=('image_feature', 'word_data'))
    data_shapes = [('image_feature', (batch_size, 4096)), ('word_data', (batch_size, 38))]
    label_shapes = [('softmax_label', (batch_size,))]
    mod.bind(data_shapes=data_shapes, label_shapes=label_shapes)
    if evalua:
        return mod
    infer_sym = caption_module(num_lstm_layer=1, seq_len=1, vocab_size=input_vocab_size, num_hidden=256, num_embed=256, batch_size=batch_size)
    infer_mod = mx.mod.Module(symbol=infer_sym, data_names=('image_feature', 'word_data'))
    data_shapes = [('image_feature', (batch_size, 4096)), ('word_data', (batch_size, 1))]
    infer_mod.bind(data_shapes=data_shapes)
    infer_mod.init_params()
    infer_mod.set_params(arg_params=mod.get_params()[0], aux_params=mod.get_params()[1])
    del mod
    return infer_mod

def Batch(image, word, target, batch_size):
    data_iter = mx.io.NDArrayIter(
        data={
            'image_feature': image,
            'word_data': word
            },
        label={'softmax_label': target},
        batch_size=batch_size
    )
    return data_iter

m = load_model(prefix='checkpoint/checkpoint', epoch=50, ctx=mx.cpu(), batch_size=1, evalua=False)
if args.eval:
    eval_dir = 'outputs'
    logging.info('score test...')
    score = []
    prob = []
    for image, prefix_group, target_group in zip(test_grouped_images, test_grouped_prefixes, test_grouped_targets):
        image_repeats = image.repeat(len(prefix_group), axis=0)
        data_batch = Batch(image=image_repeats, word=prefix_group, target=target_group, batch_size=len(prefix_group))
        score.append(m.score(data_batch, mx.metric.Perplexity(-1))[0][1])
        if args.gen_prob:
            prob.append(m.predict(data_batch, num_batch=len(test_grouped_prefixes)))
        del data_batch
        if len(score) % 100 == 0:
            logging.info('currtent samples {}, mean-score {}'.format(len(score), np.mean(score)))
    prob_np = map(lambda x: np.argmax(x.asnumpy(), axis=1), prob)

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    with open(eval_dir + '/prob.txt', 'w') as f:
        for row in prob_np:
            print(*[ str(p) for p in row ], sep='\t', file=f)
    with open(eval_dir + '/score.txt', 'w') as f:
        for row in score:
            print(str(row), sep='\t', file=f)

if args.infer:
    beam_width = 40
    clip_len = 50
    def predict_func(image, word):
        data_iter = mx.io.NDArrayIter(
        data={
            'image_feature': image,
            'word_data': word
            }
        )
        return m.predict(data_iter).asnumpy()

    captions = list()
    for i,image in enumerate(training_images[::5]):
        predictions_function = lambda prefixes:predict_func(image.repeat(len(prefixes), axis=0), prefixes)
        tokens = generate_sequence_beamsearch(predictions_function, prefix_vocabulary, target_vocabulary, beam_width, None, clip_len)[0]
        caption = ' '.join(tokens)
        captions.append(caption)
        print(captions)



