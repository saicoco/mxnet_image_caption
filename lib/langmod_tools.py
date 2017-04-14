from __future__ import absolute_import, division, print_function, unicode_literals

# from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                    #   int, map, next, oct, open, pow, range, round,
                    #   str, super, zip)

import random
import math
import heapq
import numpy as np

##################################################################################################################################
def text_to_prefixes(prefix_vocabulary, target_vocabulary, texts):
    """
    generate prefixes by text, for next input, it needs previous output
    and concanates it as next input : [word_0: word_pre] + [word_now]
    """
    if prefix_vocabulary.start_index is None:
        raise ValueError()
    if target_vocabulary.start_index is not None:
        raise ValueError()
    if target_vocabulary.end_index is None:
        raise ValueError()
    
    num_prefixes   = sum(len(text)+1 for text in texts)
    # max_prefix_len = max(len(text) for text in texts)+1
    max_prefix_len = 38
    
    text_indexes = np.empty((num_prefixes,), 'int32')
    prefixes     = np.full((num_prefixes, max_prefix_len), prefix_vocabulary.pad_index, 'int16')
    targets      = np.empty((num_prefixes,), 'int16')
    
    glb_prefix = 0
    for (i, text) in enumerate(texts):
        prefix_indexes = [ prefix_vocabulary.start_index ] + prefix_vocabulary.tokens_to_indexes(text)
        target_indexes = target_vocabulary.tokens_to_indexes(text) + [ target_vocabulary.end_index ]
        for loc_prefix in range(1, len(target_indexes)+1):
            text_indexes[glb_prefix]           = i
            prefixes[glb_prefix][-loc_prefix:] = prefix_indexes[:loc_prefix]
            targets[glb_prefix]                = target_indexes[loc_prefix-1]
            glb_prefix += 1
    
    return (text_indexes, prefixes, targets)
    
##################################################################################################################################
def text_to_prefixes_grouped(prefix_vocabulary, target_vocabulary, texts):
    if prefix_vocabulary.start_index is None:
        raise ValueError()
    if target_vocabulary.start_index is not None:
        raise ValueError()
    if target_vocabulary.end_index is None:
        raise ValueError()
    
    # max_prefix_len = max(len(text) for text in texts)+1
    max_prefix_len = 38
    grouped_prefixes = list()
    grouped_targets  = list()
    
    for text in texts:
        num_prefixes   = len(text)+1
        
        prefixes     = np.full((num_prefixes, max_prefix_len), prefix_vocabulary.pad_index, 'int16')
        targets      = np.empty((num_prefixes,), 'int16')
    
        prefix_indexes = [ prefix_vocabulary.start_index ] + prefix_vocabulary.tokens_to_indexes(text)
        target_indexes = target_vocabulary.tokens_to_indexes(text) + [ target_vocabulary.end_index ]
        for loc_prefix in range(1, len(target_indexes)+1):
            prefixes[loc_prefix-1][-loc_prefix:] = prefix_indexes[:loc_prefix]
            targets[loc_prefix-1]                = target_indexes[loc_prefix-1]
        
        grouped_prefixes.append(prefixes)
        grouped_targets.append(targets)
    
    return (grouped_prefixes, grouped_targets)

##################################################################################################################################
def generate_sequence_beamsearch(predictions_function, prefix_vocabulary, target_vocabulary, beam_width=1, unknown_token='<UNK>', clip_len=None):
    if prefix_vocabulary.start_index is None:
        raise ValueError('Cannot generate sequence with an undefined start_index in the prefix vocabulary.')
    if target_vocabulary.end_index is None:
        raise ValueError('Cannot generate sequence with an undefined end_index in the target vocabulary.')
        
    prev_beam = Beam(beam_width)
    prev_beam.add(np.array(1.0, 'float64'), False, [ prefix_vocabulary.start_index ])
    while True:
        curr_beam = Beam(beam_width)
        
        #Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
        prefix_batch = list()
        prob_batch = list()
        for (prefix_prob, complete, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                prefix_batch.append(prefix)
                prob_batch.append(prefix_prob)
            
        #Get probability of each possible next word for each incomplete prefix.
        indexes_distributions = predictions_function(prefix_batch)
        
        #Add next words
        for (prefix_prob, prefix, indexes_distribution) in zip(prob_batch, prefix_batch, indexes_distributions):
            for (next_index, next_prob) in enumerate(indexes_distribution):
                if unknown_token is None and next_index == target_vocabulary.unknown_index: #skip unknown tokens if requested
                    pass
                elif next_index == target_vocabulary.end_index: #if next word is the end token then mark prefix as complete and leave out the end token
                    curr_beam.add(prefix_prob*next_prob, True, prefix)
                else: #if next word is a non-end token then mark prefix as incomplete
                    curr_beam.add(prefix_prob*next_prob, False, prefix+[target_vocabulary.convert_index(prefix_vocabulary, next_index)])
        
        (best_prob, best_complete, best_prefix) = max(curr_beam)
        if best_complete == True or (clip_len is not None and len(best_prefix)-1 == clip_len): #if the length of the most probable prefix exceeds the clip length (ignoring the start token) then return it as is
            return (prefix_vocabulary.indexes_to_tokens(best_prefix[1:], unknown_token=unknown_token), best_prob) #return best sentence without the start token and together with its probability
            
        prev_beam = curr_beam
            
class Beam(object):
#For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
#This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    #################################################################
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    #################################################################
    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
    
    #################################################################
    def __iter__(self):
        return iter(self.heap)
                    
                    
##################################################################################################################################
def predict_next_tokens(prediction_function, prefix_vocabulary, target_vocabulary, tokens_prefix, unknown_token='<UNK>', end_token='<END>'):
    prefix = [ prefix_vocabulary.start_index ] + prefix_vocabulary.tokens_to_indexes(tokens_prefix)
            
    indexes_distribution = prediction_function(prefix)
    next_indexes = [
                    (
                        target_vocabulary.index_to_token(index, unknown_token=unknown_token, end_token=end_token),
                        prob
                    )
                    for (prob, index) in zip(indexes_distribution, range(target_vocabulary.num_all_tokens))
                ]
    next_indexes.sort(key=lambda x:x[1], reverse=True)
    return next_indexes
    
    
##################################################################################################################################
def sequence_probability(prob_seq):
    return np.prod(prob_seq, dtype=np.float64)
    
##################################################################################################################################
def sequence_perplexity(prob_seq):
    return 2.0**(sum(-np.log2(prob_seq))/len(prob_seq))
    
