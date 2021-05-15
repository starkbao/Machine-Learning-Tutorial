*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Sequence-to-Sequence Model, please watch it on [YouTube](https://www.youtube.com/watch?v=f1KUUz7v8g4&feature=youtu.be&ab_channel=Hung-yiLee).*

# Sequence-to-Sequence (Seq2Seq) Model
Seq2seq is a family of machine learning approaches used for language processing. Applications include language translation, image captioning, conversational models, and text summarization.\
Seq2seq turns one sequence into another sequence (sequence transformation). It does so by use of a **recurrent neural network (RNN)** or more often **LSTM or GRU** to avoid the problem of vanishing gradient. The context for each item is the output from the previous step. The primary components are one encoder and one decoder network. The encoder turns each item into a corresponding hidden vector containing the item and its context. The decoder reverses the process, turning the vector into an output item, using the previous output as the input context.

Optimizations include:

- Attention: The input to the decoder is a single vector that stores the entire context. **Attention allows the decoder to look at the input sequence selectively**.
- Beam Search: Instead of picking the single output (word) as the output, **multiple highly probable choices are retained, structured as a tree** (using a Softmax on the set of attention scores). Average the encoder states weighted by the attention distribution.
- Bucketing: Variable-length sequences are possible because of padding with 0s, which may be done to both input and output. However, if the sequence length is 100 and the input is just 3 items long, expensive space is wasted. Buckets can be of varying sizes and specify both input and output lengths.

Training typically uses a cross-entropy loss function, whereby one output is penalized to the extent that the probability of the succeeding output is less than 1.\
([from WikiPedia](https://en.wikipedia.org/wiki/Seq2seq))

# Task Description
In this tutorial, we'll build a Seq2Seq model to translate the sentence from English into Chinese. Also, three methods, that is, the Attention, Schedule Sampling, and Beam Seach are being investigated as well.
## The Seq2Seq Model
The model conbines two RNNs. One is an Encoder, and the other is a Decoder. The decoder transforms the input English sentence into a vector (latent representation). The Decoder transforms the output vector from the Encoder to the Chinese sentence. 
<p align="center"><img width="70%" src="https://i.imgur.com/Bhh2UuN.jpg"></p>

## Attention
Generally speaking, there are three steps that happened in the Attention:
1. Compute the "Attention Weight" from the hidden vector of the Decoder and (the hidden vector of) Encoder.
2. Compute the "Attention Vector" from the weighted sum of Attention Weight and the hidden vector of Encoder.
3. Pass the Attention Vector to the Decoder (by adding or connecting them).
<p align="center"><img width="50%" src="https://i.imgur.com/s6ovDPE.jpg"></p>

## Schedule Sampling
Schedule Sampling is to resolve the inconsistency between training and testing.\
We'll use the predicted output from the model itself to the Decoder input in a certain probability.
<p align="center"><img width="80%" src="https://i.imgur.com/hZKVS57.jpg"></p>

## Beam Search
The model will not take the highest probability as the answer every time since it may only be the local optimum instead of the global optimum.\
In practice, the exhaustive method is not feasible, so each Decoder step is fixed to take the sentence with the highest K probability of the current generated sentence.
<p align="center"><img width="60%" src="https://i.imgur.com/XnGOQlM.jpg"></p>

## Evaluation Metrics
The evaluation metric is the BLEU Score. For more information on it, please refer to [here](https://en.wikipedia.org/wiki/BLEU).

# Dataset
The dataset is from the *cmn-eng* dataset in [ManyThings.org](https://www.manythings.org/anki/).
- Training set: 18,000 sentences.
- Validation set: 500 sentences.
- Testing set: 2,636 sentence.
The format is as follows. Each sentance is separated by `TAB ('\t')`. Each word is separate by a space.\
<p align="center"><img width="60%" src="https://i.imgur.com/74GhSWa.jpg"></p>

Also, there are two dictionaries.
1. `int2word_*.json`: transforming the integer to word.
2. `word2int_*.json`: transforming the word to an integer.

The `*` sign above is `en` or `cn` for representing English and Chinese respectively.



# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [Seq2seq from Wikipedia](https://en.wikipedia.org/wiki/Seq2seq)
