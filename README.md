## image caption generation  

This is a simple implementaion of paper Neural [Image Caption][^1] based on mxnet.  
Some codes refer [where-to-image](https://github.com/mtanti/where-image)

### Usage  

1. Prepare the datasets and pre_train params to dirs `datasets` and `pre_train`, here we use pretrain-model is vgg-16, datasets are Filckr8k, you could replace it with your datasets and pretrain_model. For Flickr8k, which includes images and captions, captions are store in dataset.json, looks like following:  
```
{"images": 
    [
        {"sentids": [0, 1, 2, 3, 4], 
        "imgid": 0, 
        "sentences": [
        {"tokens": ["a", "black", "dog", "is", "running", "after", "a", "white", "dog", "in", "the", "snow"], "raw": "A black dog is running after a white dog in the snow .", "imgid": 0, "sentid": 0}, 
        {"tokens": ["black", "dog", "chasing", "brown", "dog", "through", "snow"], "raw": "Black dog chasing brown dog through snow", "imgid": 0, "sentid": 1}, 
        {"tokens": ["two", "dogs", "chase", "each", "other", "across", "the", "snowy", "ground"], "raw": "Two dogs chase each other across the snowy ground .", "imgid": 0, "sentid": 2}, 
        {"tokens": ["two", "dogs", "play", "together", "in", "the", "snow"], "raw": "Two dogs play together in the snow .", "imgid": 0, "sentid": 3}, 
        {"tokens": ["two", "dogs", "running", "through", "a", "low", "lying", "body", "of", "water"], "raw": "Two dogs running through a low lying body of water .", "imgid": 0, "sentid": 4}
                    ], 
        "split": "train", "filename": "2513260012_03d33305cf.jpg"}, ...
    
    ],
"datasets":Flickr8k}
```
or you can download processed data from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/), which image are extracted from vgg networks 4096-dim, and unzip them into dir 'datasets',
then copy file which in "old" dir into root dir, and run it, this is a old version about NIC.  


2. After data downloading completes, you can run:  
```
python 1_preprocess_data.py
```
when it runs over, there will be a directory named "processed_data" which include train, val and test datasets which are splited by "split" key in dataset.json .

3. `python 2_train_val.py` to train model on your dataset and save you dataset.  

4. There are something wrong with test stage(predict), (variable length for sym, I think I should use `mx.mod.BuckingModule`), I am trying~~~~~~~~~~~, if you find the solution, welcome to
issue me.

### Reference  
[^1]: Vinyals O, Toshev A, Bengio S, et al. Show and tell: A neural image caption generator[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3156-3164.

