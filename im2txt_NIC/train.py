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

if __name__ == '__main__':

    learning_rate = 0.01
    epoches = 20
    batch_size = 8
    num_hidden = 256
    num_embed = 256
    num_lstm_layer = 1
    ctx = mx.gpu(0)

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
        ctx=ctx, is_train=True, grad_req='add', **lstm_shapes)

    # init params
    pretrain = mx.nd.load(config.vgg_pretrain)
    init_cnn(cnn_exec, pretrain)

    # init optimazer
    optimazer = mx.optimizer.create('sgd')
    optimazer.lr = learning_rate
    updater = mx.optimizer.get_updater(optimazer)

    # init metric
    perplexity = mx.metric.Perplexity(ignore_label=-1)
    perplexity.reset()

    # callback
    callbacks = collections.namedtuple('callbacks', 'nbatch eval_metric epoch')
    params = callbacks(nbatch=len(train_data.idx)//batch_size, eval_metric=perplexity, epoch=epoches)
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

            # update lstm grad
            for key, arr in lstm_exec.grad_dict.items():
                arr[:] = 0.
            
            lstm_exec.forward(is_train=True)
            params.eval_metric.update(labels=batch.label,
                              preds=lstm_exec.outputs)

            lstm_exec.backward()
            speedometer(params)

            for j, name in enumerate(lstm.list_arguments()):
                if name not in lstm_shapes.keys():
                    updater(j, lstm_exec.grad_dict[name], lstm_exec.arg_dict[name])
    ##########################################################################

    # train_wrap = dataIter_wrap(train_data, ctx=ctx)
    # val_wrap = dataIter_wrap(val_data, ctx=ctx)

    # lstm = caption_module(num_lstm_layer=num_lstm_layer, seq_len=train_data.sent_length,
    #                       vocab_size=train_data.vocab_size, num_hidden=num_hidden, num_embed=num_embed, batch_size=batch_size)
    # caption = mx.mod.Module(symbol=lstm, data_names=['image_feature', 'word_data'], label_names=('softmax_label',), context=ctx)

    # caption.fit(
    #     train_data=train_wrap,
    #     eval_data=val_wrap,
    #     eval_metric=mx.metric.Perplexity(None),
    #     optimizer='sgd',
    #     optimizer_params={'learning_rate': 0.01,
    #                       'momentum': 0.9,
    #                       'wd': 0.00001},
    #     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
    #     num_epoch=20,
    #     batch_end_callback=mx.callback.Speedometer(batch_size, 20))
    # # caption.bind(data_shapes=caption_data_shapes, label_shapes=[
    #              ('softmax_label', (batch_size, 30))])
    # caption.init_params()
    # caption.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(
    #     ('learning_rate', 0.01),), force_init=False)

    # init sym
    # pretrain_cnn = mx.nd.load(config.vgg_pretrain)
    # init_cnn(cnn_exec, pretrain_cnn)

    # init metric,monitor invalid label:-1
    # perplexity = mx.metric.Perplexity(-1)
    # monitor = mx.callback.Speedometer(batch_size=batch_size, frequent=50)
    # caption.install_monitor(monitor)

    ###########################################################
    #################### train lopp ###########################
    ###########################################################
    # for epoch in range(1):
    #     tic = time.time()
    #     perplexity.reset()
    #     for i, batch in enumerate(train_data):
    #         # get image_feature from fc_layer
    #         # monitor.tic()
    #         cnn_exec.arg_dict['image_data'][:] = batch.data[0]
    #         cnn_exec.forward()
    #         image_feature = cnn_exec.outputs[0]

    #         # input image_feature to caption
    #         data = [image_feature, batch.data[1]]
    #         provide_data = [
    #             ('image_feature', image_feature.shape), batch.provide_data[1]]

    #         caption_batch = mx.io.DataBatch(data=data, label=batch.label,
    #                                         bucket_key=batch.bucket_key,
    #                                         provide_data=provide_data,
    #                                         provide_label=batch.provide_label)
    #         caption.forward(data_batch=caption_batch)
    #         try:
    #             caption.backward()
    #         except:
    #             logging.info('bucket-idx:{}, batch-idx:{}, cur-idx:{}, length-idx:{}'.format(
    #                 caption_batch.bucket_key, i, train_data.curr_idx, len(train_data.idx)))
    #             # caption.update()

    #     caption.update_metric(eval_metric=perplexity, labels=caption_batch.label)
    #     # monitor.toc_print()
    # # one epoch of training is finished
    # for name, val in perplexity.get_name_value():
    #     self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
    # toc = time.time()
    # self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
    # train_data.reset()
