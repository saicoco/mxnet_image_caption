# -*- conding=utf-8 -*-

import json
import mxnet as mx
import config

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


    
def vocab_generate(tokens):
    sents, tmp_vocab = mx.rnn.encode_sentences(
        sentences=tokens,
        vocab=None,
        invalid_label=-1,
        invalid_key='UNK',
        start_label=0
    )
    # filter words that low than words_cout_threshold
    vocab = {k : tmp_vocab[k] for k in tmp_vocab}
    return vocab

def load_json():
    with open(config.text_root, 'r') as f:
        dataset_json = json.load(f)
    return dataset_json

def idx2words(vocab):
    idx2word = {vocab[k] : k for k in vocab}
    return idx2word

def save2json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)
    print('save done')
if __name__ == '__main__':
    dataset = load_json()
    tokens = [dataset['images'][i]['sentences'][j]['tokens'] for i in xrange(len(dataset['images'])) for j in xrange(len(dataset['images'][i]['sentences']))]
    vocab = vocab_generate(tokens)
    idx_words = idx2words(vocab)
    save2json(idx_words, config.idx2words)
    with open(config.vocab_root, 'w') as f:
        json.dump(vocab, f)


 