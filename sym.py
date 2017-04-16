# -*- conding=uft-8 -*-

"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import mxnet as mx

def vgg16_fc7(input_name='image_data'):
    ## define alexnet
    data = mx.symbol.Variable(name=input_name)
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    return fc7

def caption_module(num_lstm_layer=1, seq_len=38, vocab_size=2540, num_hidden=256, num_embed=256, batch_size=50, dropout=0.):

    '''
    input_vocab_size: len(tokens) + 3
    output_vocab_size: len(tokens) + 2
    '''
    seq = mx.sym.Variable('word_data')
    label = mx.sym.Variable('softmax_label')
    image_feature = mx.sym.Variable('image_feature')
    image_embed = mx.sym.FullyConnected(data=image_feature, num_hidden=num_embed, name='img_embed')
    word_embed = mx.sym.Embedding(data=mx.sym.BlockGrad(seq), input_dim=vocab_size, output_dim=num_embed, name='seq_embed')

    # Concat image_embed with word_embed in axis=1 (batch_size, length+1, num_hidden)
    image_embed_reshape = mx.sym.expand_dims(image_embed, axis=1, name='image_embed_expand_dims')
    embedd_feature = mx.sym.Concat(image_embed_reshape, word_embed, num_args=2, dim=1, name='embed_concat')

    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_lstm_layer):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_' % i))
    outputs, states = stack.unroll(length=seq_len+1, inputs=embedd_feature, merge_outputs=False)
    pred = mx.sym.FullyConnected(outputs[-1], num_hidden=vocab_size-1, name='pred_fc')
    label = mx.sym.Reshape(label, shape=(-1,))
    softmax_output = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    softmax_output = mx.sym.SoftmaxOutput(data=pred, name='softmax')
    return softmax_output


if __name__ == '__main__':
    lstm = caption_module(num_lstm_layer=1, seq_len=1, vocab_size=2540, num_hidden=256, num_embed=256, batch_size=50)
    lstm_exec = lstm.simple_bind(ctx=mx.cpu(0), is_train=False, word_data=(12, 1),image_feature=(12, 4096))
    print lstm.infer_shape(word_data=(50, 1),  image_feature=(50, 4096))
    
