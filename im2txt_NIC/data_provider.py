# -*- conding=uft-8 -*-

"""
flikcr8k dataset provider
"""
import mxnet as mx
import numpy as np
import bisect
import config
import json
import random
import scipy
from model import vgg16_fc7


class variable_length_caption_dataIter(mx.io.DataIter):
    """Simple bucketing iterator for caption model.
    Label for each step is constructed from data of
    next step.

    Parameters
    ----------
    captions : list of list of tokens and image name
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    mode: str, default 'train'
        train or test, val
    """

    def __init__(self, captions, batch_size, buckets=None, invalid_label=-1,
                 data_name=['image_data', 'word_data'], label_name='softmax_label', dtype='float32', mode='train'):
        super(caption_dataIter, self).__init__()

        assert mode in [u'train', u'val', u'test']
        if not buckets:
            all_sentences = [captions['images'][i]['sentences'][j]['tokens'] for i in xrange(len(captions['images']))
                             for j in xrange(len(captions['images'][i]['sentences']))]
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in all_sentences]))
                       if j >= batch_size]
        ndiscard = 0
        self.words_data = [[] for _ in buckets]
        self.words_buckets = buckets
        self.image_data = [[] for _ in buckets]
        self.vocab_size = None
        with open(config.vocab_root, 'r') as f:
            vocab = json.load(f)
            self.vocab_size = len(vocab)
            for i in range(len(captions['images'])):
                if captions['images'][i]['split'] == mode:
                    sentences = [captions['images'][i]['sentences'][j]['tokens']
                                 for j in xrange(len(captions['images'][i]['sentences']))]
                    for k in range(len(sentences)):
                        buck = bisect.bisect_left(
                            self.words_buckets, len(sentences[k]))
                        if buck == len(self.words_buckets):
                            ndiscard += 1
                            continue
                        buff = np.full(
                            (self.words_buckets[buck],), invalid_label, dtype)
                        buff[:len(sentences[k])] = [vocab[item]
                                                    for item in sentences[k]]
                        self.words_data[buck].append(buff)
                        self.image_data[buck].append(
                            captions['images'][i]['filename'])
            print(
                "WARNING: discarded %d sentences longer than the largest bucket." % ndiscard)
            self.words_data = [np.asarray(i, dtype=dtype)
                               for i in self.words_data]

        self.batch_size = batch_size
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.ndword = []
        self.ndlabel = []
        self.default_bucket_key = max(self.words_buckets)

        self.provide_data = [(data_name[0], (batch_size, 3, 224, 224)), (data_name[
            1], (batch_size, self.default_bucket_key))]
        self.provide_label = [
            (label_name, (batch_size, self.default_bucket_key))]

        self.idx = []
        for i, buck in enumerate(self.words_data):
            self.idx.extend([(i, j) for j in range(
                0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        self.curr_idx = 0
        # random.shuffle(self.idx)

        self.ndlabel = []
        self.ndword = []
        for buck in self.words_data:
            label = np.empty_like(buck)
            label[:, :-1] = buck[:, 1:]
            label[:, -1] = self.invalid_label
            self.ndword.append(mx.nd.array(buck, dtype=self.dtype))
            self.ndlabel.append(mx.nd.array(label, dtype=self.dtype))

    def one_hot(self, arr):
        one_hot_arr = np.zeros(shape=(len(arr), self.vocab_size))
        for i, item in enumerate(arr):
            one_hot_arr[i][item] = 1.0
        return one_hot_arr

    def next(self):
        if self.curr_idx == len(self.idx):

            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1
        word_data = self.ndword[i][j:j + self.batch_size]
        label = self.ndlabel[i][j:j + self.batch_size]
        image_name = self.image_data[i][j:j + self.batch_size]

        # hot_label = [self.one_hot(label[i]) for i in range(label.shape[0])]
        # hot_label = mx.nd.array(hot_label, dtype=self.dtype)
        image_data = []
        for item in image_name:
            filename = config.image_root + item
            img = scipy.misc.imread(filename)
            img = scipy.misc.imresize(img, size=(224, 224))
            # img = cv2.imread(filename)
            # img = cv2.resize(img, dsize=(224, 224))
            img = np.transpose(img, (2, 0, 1))
            image_data.append(img)
        image_data = mx.nd.array(image_data, dtype=self.dtype)

        return mx.io.DataBatch([image_data, word_data], [label],
                               bucket_key=self.words_buckets[i],
                               provide_data=[(self.data_name[0], image_data.shape),
                                             (self.data_name[1], word_data.shape)],
                               provide_label=[(self.label_name, label.shape)])

class caption_dataIter(mx.io.DataIter):
    """Simple bucketing iterator for caption model.
    Label for each step is constructed from data of
    next step.

    Parameters
    ----------
    captions : list of list of tokens and image name
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    mode: str, default 'train'
        train or test, val
    """

    def __init__(self, captions, batch_size, buckets=None, invalid_label=-1,
                 data_name=['image_data', 'word_data'], label_name='softmax_label', dtype='float32', mode='train'):
        super(caption_dataIter, self).__init__()

        assert mode in [u'train', u'val', u'test']
        all_sentences_length = [len(captions['images'][i]['sentences'][j]['tokens']) for i in xrange(len(captions['images']))
                             for j in xrange(len(captions['images'][i]['sentences']))]
        max_length = max(all_sentences_length)
        self.sent_length = max_length
        self.words_data = []
        self.image_data = []
        self.vocab_size = None
        with open(config.vocab_root, 'r') as f:
            vocab = json.load(f)
            self.vocab_size = len(vocab)
            for i in range(len(captions['images'])):
                if captions['images'][i]['split'] == mode:
                    sentences = [captions['images'][i]['sentences'][j]['tokens']
                                 for j in xrange(len(captions['images'][i]['sentences']))]
                    for k in range(len(sentences)):
                        buff = np.full(
                            (self.sent_length,), invalid_label, dtype)
                        buff[:len(sentences[k])] = [vocab[item]
                                                    for item in sentences[k]]
                        self.words_data.append(buff)
                        self.image_data.append(
                            captions['images'][i]['filename'])

            self.words_data = np.asarray(self.words_data, dtype=dtype)

        self.batch_size = batch_size
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.ndword = []
        self.ndlabel = []

        self.provide_data = [(data_name[0], (batch_size, 3, 224, 224)), (data_name[
            1], (batch_size, self.sent_length))]
        self.provide_label = [
            (label_name, (batch_size, self.sent_length))]

        self.idx = range(len(self.words_data) - self.batch_size + 1)
        self.curr_idx = 0
        self.reset()

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)

        self.ndlabel = []
        self.ndword = mx.nd.array(self.words_data[self.idx], dtype=self.dtype)

        label = np.empty_like(self.words_data)
        label[:, :-1] = self.words_data[:, 1:]
        label[:, -1] = self.invalid_label
        self.ndlabel = mx.nd.array(label, dtype=self.dtype)


    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        self.curr_idx += 1
        word_data = self.ndword[self.curr_idx:self.curr_idx + self.batch_size]
        label = self.ndlabel[self.curr_idx:self.curr_idx + self.batch_size]
        image_name = self.image_data[self.curr_idx:self.curr_idx + self.batch_size]

        image_data = []
        for item in image_name:
            filename = config.image_root + item
            img = scipy.misc.imread(filename)
            img = scipy.misc.imresize(img, size=(224, 224))
            img = np.transpose(img, (2, 0, 1))
            image_data.append(img)
        image_data = mx.nd.array(image_data, dtype=self.dtype)

        return mx.io.DataBatch([image_data, word_data], [label],
                               provide_data=[(self.data_name[0], image_data.shape),
                                             (self.data_name[1], word_data.shape)],
                               provide_label=[(self.label_name, label.shape)])


class dataIter_wrap(mx.io.DataIter):

    def __init__(self, dataIter, ctx=mx.cpu()):
        super(dataIter_wrap, self).__init__()
        cnn_shape = dict(dataIter.provide_data[:1])
        cnn_sym = vgg16_fc7(input_name='image_data')
        cnn_exec = cnn_sym.simple_bind(ctx=ctx, is_train=False, **cnn_shape)
        pretrain_cnn = mx.nd.load(config.vgg_pretrain)
        init_cnn(cnn_exec, pretrain_cnn)
        self.cnn_exec = cnn_exec
        self.dataIter = dataIter
        self.provide_data = [('image_feature', (dataIter.batch_size, 4096)),
                            dataIter.provide_data[1]]
        self.provide_label = dataIter.provide_label

    def reset(self):
        self.dataIter.reset()
    
    def image_feature_generator(self):
        for i, batch in enumerate(self.dataIter):
            self.cnn_exec.arg_dict['image_data'][:] = batch.data[0]
            self.cnn_exec.forward()
            image_feature = self.cnn_exec.outputs[0]

            # input image_feature to caption
            data = [image_feature, batch.data[1]]
            provide_data = [('image_feature', image_feature.shape),
                            batch.provide_data[1]]
            caption_batch = mx.io.DataBatch(data=data, label=batch.label,
                                            provide_data=provide_data,
                                            provide_label=batch.provide_label)
            yield caption_batch
    
    def next(self):
        return self.image_feature_generator().next()

def init_cnn(cnn, pretrain):
    for k, arr in cnn.arg_dict.items():
        if k == 'image_data':
            continue
        arr[:] = pretrain['arg:' + k][:]

if __name__ == '__main__':
    with open(config.text_root, 'r') as f:
        captions = json.load(f)
    diter = caption_dataIter(captions=captions, batch_size=100)
    wrap_iter = dataIter_wrap(diter, ctx=mx.gpu(1))
    # data = wrap_iter.next()
    # sent =  data.data[1][0].asnumpy()
    # print sent.shape
    for i, batch in enumerate(wrap_iter):
        print i
    # data = diter.next()
    # image, sent, label = data.data[0][0].asnumpy(), data.data[1][
    #     0].asnumpy(), data.label[0][0].asnumpy()
    # print label.shape, diter.provide_data, diter.provide_label

    # with open(config.idx2words, 'r') as f:
    #     idx2words = json.load(f)
    # if 0 in idx2words.keys():
    #     print 'yes'
    # sentence = [idx2words[str(int(k))] for k in sent]
    # print sentence
    # label_sent = [idx2words[str(int(k))] for k in label]
    # print sentence, diter.default_bucket_key, label_sent

