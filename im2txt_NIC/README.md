## image caption generation  

This is a simple implementaion of paper Neural [Image Caption][^1] based on mxnet.  

### Usage  

1. Prepare the datasets and pre_train params to dirs `datasets` and `pre_train`, here we use pretrain-model is vgg-16, datasets are Filckr8k, you could replace it with your datasets and pretrain_model. For Flickr8k, which includes images and captions, captions are store in dataset.json, looks like following:  
```
{"images": [{"sentids": [0, 1, 2, 3, 4], "imgid": 0, "sentences": [{"tokens": ["a", "black", "dog", "is", "running", "after", "a", "white", "dog", "in", "the", "snow"], "raw": "A black dog is running after a white dog in the snow .", "imgid": 0, "sentid": 0}, {"tokens": ["black", "dog", "chasing", "brown", "dog", "through", "snow"], "raw": "Black dog chasing brown dog through snow", "imgid": 0, "sentid": 1}, {"tokens": ["two", "dogs", "chase", "each", "other", "across", "the", "snowy", "ground"], "raw": "Two dogs chase each other across the snowy ground .", "imgid": 0, "sentid": 2}, {"tokens": ["two", "dogs", "play", "together", "in", "the", "snow"], "raw": "Two dogs play together in the snow .", "imgid": 0, "sentid": 3}, {"tokens": ["two", "dogs", "running", "through", "a", "low", "lying", "body", "of", "water"], "raw": "Two dogs running through a low lying body of water .", "imgid": 0, "sentid": 4}], "split": "train", "filename": "2513260012_03d33305cf.jpg"}, ...],
"datasets":Flickr8k}
```
For pretrain weight, you can download from [here](http://data.dmlc.ml/mxnet/models/imagenet/)


2. Use tools.py to generate vocab.json and idx2word.json that will ba save into dir 'vocab'. 

3. `python train.py` to train model on your dataset  

4. to be continue... 

### Reference  
[1]: Vinyals O, Toshev A, Bengio S, et al. Show and tell: A neural image caption generator[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3156-3164.

