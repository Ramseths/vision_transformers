# Transformers

1. The image is divided into a grid of patches of the same size.
2. Each patch is converted to a vector by a linear convolution layer.
3. Position information is added to each patch vector.
4. The patch vectors are used as input to a 1D sequence.
5. The sequence is fed to the Transformer encoder, which has multiple layers of attention and forward feed.
6. The encoder learns image patterns and features through the attention and forward feed.
7. The output of the encoder is used as input to the decoder.
8. The decoder uses an attention technique called mask attention to prevent access to future information and autoregressive decoding to perform classification.
9. The output of the decoder is a probability distribution over the possible classes.
10. The class with the highest probability is selected as the network prediction.

# Project

The Vision Transformer employs the Transformer Encoder that was proposed in the [attention is all you need paper](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

