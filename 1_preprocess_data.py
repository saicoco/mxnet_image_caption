# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import scipy.io as sio
from lib.vocabulary import *
import json
from lib.langmod_tools import *
import os

raw_data_dir = 'datasets/flickr8k'
processed_input_data_dir = 'processed_data'
min_token_freq = 5

if not os.path.exists(processed_input_data_dir):
    os.mkdir(processed_input_data_dir)

print('load data...')
with open(raw_data_dir + '/dataset.json', 'r') as f:
    captions_data = json.load(f)['images']
image_features = sio.loadmat(raw_data_dir + '/vgg_feats.mat')['feats'].T
raw_dataset = {
    'train': {'filenames': list(), 'images': list(), 'captions': list()},
    'val':   {'filenames': list(), 'images': list(), 'captions': list()},
    'test':  {'filenames': list(), 'images': list(), 'captions': list()},
}
for (image_id, (caption_data, image_fea)) in enumerate(zip(captions_data, image_features)):
    assert caption_data['sentences'][0]['imgid'] == image_id
    split = caption_data['split']
    filename = caption_data['filename']
    caption_group = [caption['tokens']
                     for caption in caption_data['sentences']]
    image = image_fea / np.linalg.norm(image_fea)

    for caption in caption_group:
        raw_dataset[split]['filenames'].append(filename)
        raw_dataset[split]['images'].append(image)
        raw_dataset[split]['captions'].append(caption)

print('process data...')
all_tokens = (token for caption in raw_dataset[
              'train']['captions'] for token in caption)
tokens = select_vocab_tokens(all_tokens, min_token_freq=min_token_freq)
prefix_vocabulary = Vocabulary(
    tokens, pad_index=0, start_index=1, unknown_index=-1)
target_vocabulary = Vocabulary(tokens, end_index=0, unknown_index=-1)

training_indexes, training_prefixes, training_targets = text_to_prefixes(prefix_vocabulary, target_vocabulary, raw_dataset['train']['captions'])
validation_indexes, validation_prefixes, validation_targets = text_to_prefixes(prefix_vocabulary, target_vocabulary, raw_dataset['val']['captions'])
test_grouped_prefixes, test_grouped_targets = text_to_prefixes_grouped(prefix_vocabulary, target_vocabulary, raw_dataset['test']['captions'])

training_images     = np.array(raw_dataset['train']['images'])
validation_images   = np.array(raw_dataset['val']['images'])
test_grouped_images = [ np.array([image]) for image in raw_dataset['test']['images'] ]

print('save preprocess data...')
with open(processed_input_data_dir+'/vocabulary.txt', 'w') as f:
    for token in tokens:
        print(str(token), file=f)

with open(processed_input_data_dir+'/test_humancaptions.txt', 'w') as f:
    for caption in raw_dataset['test']['captions']:
        print(str(' '.join(caption)), file=f)


known_token_set = set(tokens)
with open(processed_input_data_dir+'/info.txt', 'w') as f:
    print('Known token types:', str(len(tokens)), sep='\t', file=f)
    print('', file=f)
    print('Number of training captions:', str(sum(1 for caption in raw_dataset['train']['captions'])), sep='\t', file=f)
    print('Shortest training caption:', str(min(len(caption) for caption in raw_dataset['train']['captions'])), sep='\t', file=f)
    print('Longest training caption:', str(max(len(caption) for caption in raw_dataset['train']['captions'])), sep='\t', file=f)
    print('Unknown token types in training captions:', str(len(set(token for caption in raw_dataset['train']['captions'] for token in caption if token not in known_token_set))), sep='\t', file=f)
    print('', file=f)
    print('Number of validation captions:', str(sum(1 for caption in raw_dataset['val']['captions'])), sep='\t', file=f)
    print('Shortest validation caption:', str(min(len(caption) for caption in raw_dataset['val']['captions'])), sep='\t', file=f)
    print('Longest validation caption:', str(max(len(caption) for caption in raw_dataset['val']['captions'])), sep='\t', file=f)
    print('Unknown token types in validation captions:', str(len(set(token for caption in raw_dataset['val']['captions'] for token in caption if token not in known_token_set))), sep='\t', file=f)
    print('', file=f)
    print('Number of test captions:', str(sum(1 for caption in raw_dataset['test']['captions'])), sep='\t', file=f)
    print('Shortest test caption:', str(min(len(caption) for caption in raw_dataset['test']['captions'])), sep='\t', file=f)
    print('Longest test caption:', str(max(len(caption) for caption in raw_dataset['test']['captions'])), sep='\t', file=f)
    print('Unknown token types in test captions:', str(len(set(token for caption in raw_dataset['test']['captions'] for token in caption if token not in known_token_set))), sep='\t', file=f)
    print('', file=f)
    print('Training set size:', str(len(training_prefixes)), sep='\t', file=f)
    print('Validation set size:', str(len(validation_prefixes)), sep='\t', file=f)
    print('Test set size:', str(sum(len(group) for group in test_grouped_prefixes)), sep='\t', file=f)

with open(processed_input_data_dir+'/training_prefixes.npy', 'wb') as f:
    np.save(f, training_prefixes)
with open(processed_input_data_dir+'/training_targets.npy', 'wb') as f:
    np.save(f, training_targets)
with open(processed_input_data_dir+'/training_indexes.npy', 'wb') as f:
    np.save(f, training_indexes)
with open(processed_input_data_dir+'/training_images.npy', 'wb') as f:
    np.save(f, training_images)

with open(processed_input_data_dir+'/validation_prefixes.npy', 'wb') as f:
    np.save(f, validation_prefixes)
with open(processed_input_data_dir+'/validation_targets.npy', 'wb') as f:
    np.save(f, validation_targets)
with open(processed_input_data_dir+'/validation_indexes.npy', 'wb') as f:
    np.save(f, validation_indexes)
with open(processed_input_data_dir+'/validation_images.npy', 'wb') as f:
    np.save(f, validation_images)

with open(processed_input_data_dir+'/test_grouped_prefixes.npy', 'wb') as f:
    np.save(f, test_grouped_prefixes)
with open(processed_input_data_dir+'/test_grouped_targets.npy', 'wb') as f:
    np.save(f, test_grouped_targets)
with open(processed_input_data_dir+'/test_grouped_images.npy', 'wb') as f:
    np.save(f, test_grouped_images)
with open(processed_input_data_dir+'/test_image_filenames.txt', 'w') as f:
    print('\n'.join(raw_dataset['test']['filenames'][::5]), file=f)

