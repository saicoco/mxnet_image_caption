from __future__ import absolute_import, division, print_function, unicode_literals

# from builtins import (ascii, bytes, chr, dict, filter, hex, input,
#                      int, map, next, oct, open, pow, range, round,
#                      str, super, zip)

import numpy as np
import collections

##########################################################################


class Vocabulary:
    '''
    Stores a vocabulary of tokens that sequences can contain.
    Converts between tokens and indexes.

    Apart from the given tokens, there are 4 control tokens that have a special meaning in the library:
        pad <PAD>    : used to pad sequences to make equal length
        start <BEG>  : used to mark the beginning of a sequence
        end <END>    : used to mark the end of a sequence
        unknown <UNK>: used to mark unknown tokens that are not in the vocabulary
    Control tokens do not have an associated string and are only used as indexes
    Control tokens are all optional and can be set to None in order to have less indexes to process.
    Control tokens which are not set to None must be given indexes that come before or after those of the given tokens.
        Non-negative indexes are placed before the given tokens whilst negative indexes are placed after the given tokens.
        The mapping between tokens and indexes is as follows:
            ===============+==============+==================
                 TOKEN     | ACTUAL INDEX | CONSTRUTOR INDEX
            ===============+==============+==================
             pre-ctrl_1    | 0            | 0
             pre-ctrl_2    | 1            | 1
             ...           | ...          | ...
             pre-ctrl_n-1  | n-2          | n-2
             pre-ctrl_n    | n-1          | n-1
            ---------------+--------------+------------------
             token_1       | n            |
             token_2       | n+1          |
             ...           | ...          |
             token_m-1     | n+m-2        |
             token_m       | n+m-1        |
            ---------------+--------------+------------------
             post-ctrl_1   | n+m          | -o
             post-ctrl_2   | n+m+1        | -(o-1)
             ...           | ...          | ...
             post-ctrl_o-1 | n+m+o-2      | -2
             post-ctrl_o   | n+m+o-1      | -1
    '''

    ##########################################################################

    #################################################################
    def __init__(self, given_vocab, pad_index=None, start_index=None, end_index=None, unknown_index=None):
        '''
        Vocabulary constructor.
        Args:
            given_vocab: list of given tokens.
        '''
        non_none_controls = [index for index in [
            pad_index, start_index, end_index, unknown_index] if index is not None]
        nonneg_indexes = sorted(
            [index for index in non_none_controls if index >= 0])
        neg_indexes = sorted(
            [index for index in non_none_controls if index < 0], reverse=True)

        # check that control indexes are unique
        if len(set(non_none_controls)) != len(non_none_controls):
            raise ValueError(
                'Control indexes must be unique (indexes used: "{}")'.format(non_none_controls))

        # check that indexes are all edge indexes
        if nonneg_indexes != list(range(len(nonneg_indexes))):
            raise ValueError(
                'Non-negative control indexes must start with 0 and be contiguous e.g. [0, 1] but not [1, 2] (not start with 0) or [0, 2] (not contiguous) (non-negative indexes used: "{}")'.format(nonneg_indexes))
        if neg_indexes != list(range(-1, -len(neg_indexes) - 1, -1)):
            raise ValueError(
                'Negative control indexes must start with -1 and be contiguous e.g. [-1, -2] but not [-2, -3] (not start with -1) or [-1, -3] (not contiguous) (negative indexes used: "{}")'.format(neg_indexes))

        self.given_vocab = given_vocab
        self.inverted_given_vocab = {token: index for (
            index, token) in enumerate(self.given_vocab)}

        self.num_pre_controls = len(nonneg_indexes)
        self.num_post_controls = len(neg_indexes)
        self.num_controls = self.num_pre_controls + self.num_post_controls
        self.num_given_tokens = len(self.given_vocab)
        self.num_all_tokens = self.num_controls + self.num_given_tokens
        self.last_index = self.num_all_tokens - 1

        index_normaliser = lambda index: index if index is None or index >= 0 else self.num_all_tokens + index
        self.pad_index = index_normaliser(pad_index)
        self.start_index = index_normaliser(start_index)
        self.end_index = index_normaliser(end_index)
        self.unknown_index = index_normaliser(unknown_index)

    ##########################################################################

    #################################################################
    def is_token_known(self, token):
        '''
        '''
        return token in self.inverted_given_vocab

    #################################################################
    def token_to_index(self, token):
        '''
        '''
        if token in self.inverted_given_vocab:
            return self.num_pre_controls + self.inverted_given_vocab[token]
        else:
            if self.unknown_index is not None:
                return self.unknown_index
            else:
                raise ValueError(
                    'Unknown token "{}" used but vocabulary does not allow unknowns'.format(token))

    #################################################################
    def index_to_token(self, index, pad_token='<PAD>', start_token='<BEG>', end_token='<END>', unknown_token='<UNK>'):
        '''
        '''
        if self.num_pre_controls <= index < self.num_pre_controls + self.num_given_tokens:
            return self.given_vocab[index - self.num_pre_controls]
        else:
            token = {
                self.pad_index:     pad_token,
                self.start_index:   start_token,
                self.end_index:     end_token,
                self.unknown_index: unknown_token
            }.get(index)
            if token is None:
                raise ValueError(
                    'Non-existent index used to get token (index:{}, must be between 0 and {})'.format(index, self.last_index))
            else:
                return token

    ##########################################################################

    #################################################################
    def tokens_to_indexes(self, tokens):
        '''
        '''
        return [self.token_to_index(token) for token in tokens]

    #################################################################
    def indexes_to_tokens(self, indexes, pad_token='<PAD>', start_token='<BEG>', end_token='<END>', unknown_token='<UNK>'):
        '''
        '''
        return [self.index_to_token(index, pad_token, start_token, end_token, unknown_token) for index in indexes]

    ##########################################################################

    #################################################################
    def convert_index(self, target_vocabulary, index):
        '''
        '''
        if self.num_pre_controls <= index < self.num_pre_controls + self.num_given_tokens:
            return index - self.num_pre_controls + target_vocabulary.num_pre_controls
        else:
            try:
                new_index = {
                    self.pad_index:     target_vocabulary.pad_index,
                    self.start_index:   target_vocabulary.start_index,
                    self.end_index:     target_vocabulary.end_index,
                    self.unknown_index: target_vocabulary.unknown_index,
                }[index]
                if new_index is None:
                    raise ValueError('Attempting to convert a control token which is not present in target vocabulary (index:{}, token:{})'.format(
                        index, {self.pad_index: 'PAD', self.start_index: 'BEG', self.end_index: 'END', self.unknown_index: 'UNK'}[index]))
                else:
                    return new_index
            except KeyError:
                raise ValueError(
                    'Non-existent index used to get token (index:{}, must be between 0 and {})'.format(index, self.last_index))

    #################################################################
    def convert_indexes(self, target_vocabulary, indexes):
        '''
        '''
        return [self.convert_index(target_vocabulary, index) for index in indexes]


##########################################################################
def select_vocab_tokens(token_seq, min_token_freq=None, max_vocab_size=None):
    '''
    '''
    token_freqs = collections.Counter(token_seq)
    vocab = sorted(token_freqs.keys(),
                   key=lambda token: (-token_freqs[token], token))

    if max_vocab_size is not None:
        if max_vocab_size == 0:
            return []
        else:
            vocab = vocab[:max_vocab_size]

    if min_token_freq is not None:
        if token_freqs[vocab[-1]] >= min_token_freq:
            pass
        elif token_freqs[vocab[0]] < min_token_freq:
            vocab = []
        else:
            for glb_prefix in range(len(vocab) - 1, -1, -1):
                if token_freqs[vocab[glb_prefix]] >= min_token_freq:
                    vocab = vocab[:glb_prefix + 1]
                    break

    return vocab
