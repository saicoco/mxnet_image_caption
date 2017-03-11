# -*- conding=utf-8 -*-

import json
import mxnet as mx
import config
import time
"""
    dataset.json format:
    two keys: images, datasets
        * images: a list whose items are dict
        * dict item: keys:[imagid, raw, sentid, tokens]
"""
# def build_vocab(sentences, words_threshold=5):
#     counts = {}
#     for sentence in sentences:
#         for item in sentence:
#             counts[item] = counts.get(item, 0) + 1

#     print 'length of vocab:{}'.format(len(counts))
#     vocab = {k : counts[k] for k in counts if counts[k] > words_threshold}
#     bad_words = {k : counts[k] for k in counts if counts[k] <= words_threshold}
#     print "good words:{}, bad words:{}".format(len(vocab), len(bad_words))


# def vocab_generate(tokens):
#     sents, tmp_vocab = mx.rnn.encode_sentences(
#         sentences=tokens,
#         vocab=None,
#         invalid_label=-1,
#         invalid_key='UNK',
#         start_label=0
#     )
#     # filter words that low than words_cout_threshold
#     vocab = {k : tmp_vocab[k] for k in tmp_vocab}
#     return vocab

def load_json():
    with open(config.text_root, 'r') as f:
        dataset_json = json.load(f)
    return dataset_json


def idx2words(vocab):
    idx2word = {vocab[k]: k for k in vocab}
    return idx2word


def save2json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)
    print('save done')


def vocab_generate(sentence_iterator, word_count_threshold):
    t0 = time.time()
    word_counts = {}
    for sent in sentence_iterator:
        for w in sent:
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print "filtered words from {} to {} in {}s".format(len(word_counts), len(vocab), time.time() - t0)

    idx2word = {}
    idx2word[-1] = 'UNK'
    word2idx = {}
    word2idx['#START#'] = 0
    ix = 0
    for w in vocab:
        word2idx[w] = ix
        idx2word[ix] = w
        ix += 1
    word2idx['UNK'] = -1
    word_counts['.'] = len(sentence_iterator)
    return vocab, word2idx, idx2word
    
    
if __name__ == '__main__':
    dataset = load_json()
    tokens = [dataset['images'][i]['sentences'][j]['tokens'] for i in xrange(
        len(dataset['images'])) for j in xrange(len(dataset['images'][i]['sentences']))]
    vocab, word2idx, idx2word = vocab_generate(tokens, word_count_threshold=2)
    save2json(vocab, './vocab/vocab.json')
    save2json(word2idx, './vocab/word2idx.json')
    save2json(idx2word, './vocab/idx2word.json')
    # vocab = vocab_generate(tokens)
    # idx_words = idx2words(vocab)
    # save2json(idx_words, config.idx2words)
    # with open(config.vocab_root, 'w') as f:
    #     json.dump(vocab, f)
