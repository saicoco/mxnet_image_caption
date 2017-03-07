# -*- conding=utf-8 -*-

"""
train module
"""
import mxnet as mx
import numpy as np
import json
import config
import logging
import time
import collections
from model import vgg16_fc7, caption_module
from data_provider import caption_dataIter, init_cnn

logging.basicConfig(level=logging.INFO)


class callbacks:

    def __init__(self, nbatch, eval_metric, epoch):
        self.nbatch = nbatch
        self.eval_metric = eval_metric
        self.epoch = epoch
    

def main():
    learning_rate = 0.001
    epoches = 20
    batch_size = 25
    num_hidden = 256
    num_embed = 256
    num_lstm_layer = 2
    freq_val = 10
    val_flag = True
    ctx = mx.cpu(0)

    with open(config.text_root, 'r') as f:
        captions = json.load(f)
    buckets = [10, 20, 30]
    # buckets = None
    train_data = caption_dataIter(
        captions=captions, batch_size=batch_size, mode='train')
    val_data = caption_dataIter(
        captions=captions, batch_size=batch_size, mode='val')

    ##########################################################################
    ########################### custom train process #########################
    ##########################################################################

    cnn_shapes = {
        'image_data': (batch_size, 3, 224, 224)
    }
    cnn_sym = vgg16_fc7('image_data')
    cnn_exec = cnn_sym.simple_bind(ctx=ctx, is_train=False, **cnn_shapes)
    lstm = caption_module(num_lstm_layer=num_lstm_layer, seq_len=train_data.sent_length,
                          vocab_size=train_data.vocab_size, num_hidden=num_hidden, num_embed=num_embed, batch_size=batch_size)
    lstm_shapes = {
        'image_feature': (batch_size, 4096),
        'word_data': (batch_size, train_data.sent_length),
        'softmax_label': (batch_size, train_data.sent_length)
    }

    lstm_exec = lstm.simple_bind(
        ctx=ctx, is_train=True, **lstm_shapes)

    # init params
    pretrain = mx.nd.load(config.vgg_pretrain)
    init_cnn(cnn_exec, pretrain)

    # init optimazer
    optimazer = mx.optimizer.create('adam')
    optimazer.lr = learning_rate
    updater = mx.optimizer.get_updater(optimazer)

    # init metric
    perplexity = mx.metric.Perplexity(ignore_label=-1)
    perplexity.reset()

    # callback
    params = callbacks(nbatch=0, eval_metric=perplexity, epoch=0)
    speedometer = mx.callback.Speedometer(batch_size=batch_size, frequent=20)
    for epoch in range(epoches):
        for i, batch in enumerate(train_data):

            # cnn forward, get image_feature
            cnn_exec.arg_dict['image_data'] = batch.data[0]
            cnn_exec.forward()
            image_feature = cnn_exec.outputs[0]

            # lstm forward
            lstm_exec.arg_dict['image_feature'] = image_feature
            lstm_exec.arg_dict['word_data'] = batch.data[1]
            lstm_exec.arg_dict['softmax_label'] = batch.label

            lstm_exec.forward(is_train=True)
            params.eval_metric.update(labels=batch.label,
                              preds=lstm_exec.outputs)
            lstm_exec.backward()
            params.epoch = epoch
            params.nbatch += 1
            speedometer(params)
            for j, name in enumerate(lstm.list_arguments()):
                if name not in lstm_shapes.keys():
                    updater(j, lstm_exec.grad_dict[
                            name], lstm_exec.arg_dict[name])
        train_data.reset()
        params.nbatch = 0

        if val_flag and epoch % freq_val == 0:
            for i, batch in enumerate(val_data):

                # cnn forward, get image_feature
                cnn_exec.arg_dict['image_data'] = batch.data[0]
                cnn_exec.forward()
                image_feature = cnn_exec.outputs[0]

                # lstm forward
                lstm_exec.arg_dict['image_feature'] = image_feature
                lstm_exec.arg_dict['word_data'] = batch.data[1]
                lstm_exec.arg_dict['softmax_label'] = batch.label

                lstm_exec.forward(is_train=False)
                params.eval_metric.update(labels=batch.label,
                                preds=lstm_exec.outputs)
                params.epoch = epoch
                params.nbatch += 1
                speedometer(params)
            params.nbatch = 0
            val_data.reset()

if __name__ == '__main__':
    main()
