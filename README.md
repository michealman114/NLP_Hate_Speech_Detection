# Implementation and Evaluation of Deep Context-Aware Models for Hate Speech Detection
This repository contains code implementing and evalutating the robustness of architectures described in the paper [Detecting Online Hate Speech Using Context Aware Models](https://arxiv.org/pdf/1710.07395.pdf) as well as some extensions that we've been working on recently.

The purpose of these architectures is to take an internet comment and the title of the article it was written on and detect if the comment qualified as hate speech. The comments and titles are converted into vector embeddings with either Word2Vec (from Google) that are passed into parallel LSTM networks with attention. The concatenated outputs from the LSTM networks are then used to do a binary classification on whether the comment should be classified as hate speech or not.

This code was originally written for Google Colab to take advantage of GPU acceleration. It is currently being updated for clarity and some added functionality. Lately we've been working on using BERT to create embeddings (instead of Word2Vec) and fine tuning BERT for classification directly.

You can read our writeup on the original iteration of this project [here!](https://www.mit.edu/~anugrah/files/FinalProjectReport6864.pdf)


# Recent Results
Some new results - we reevaluated some of the cleaned models with the different pre-generated embeddings

**BERT Embeddings Results (from a lazy training loop on 30 epochs)**
- base model: Loss: 858.4895858764648
- bidirectional LSTM: Loss: 341.25000190734863
- bidirectional LSTM with Attention: Loss: 0.0014144671586109325

These results are both intuitive and surprising. It is amsuingly surprising that the LSTMS without attention are as ludicrously terrible as they are, but it kind of makes sense.  The enormous performance jump suggests that the attention mechanism (which in this application is just a very simple set of FC layers) is doing most of the work (even without being attention masked - which is something we need to fix pretty urgently). I'd be willing to bet that when working on BERT embeddings we can just trivially slap on a couple linear layers on top and get really good performance.

Also, when we were cleaning up the dataset before feeding into word2vec we originally did some classical stuff (removing stopwords, punctuation etc) that removes important embedding context - especailly since some stopwords like "no","never","not" substantially change the meaning of a sentence - which probably also explains a good amount of why BERT performs so much better. Fixing this really obvious data processing mistake isn't too difficult, but we can do that later.

**BERT Embeddings 10-fold cross validation**
- 30 epochs: bidirectional model with attention:
    - Accuracy: 0.7532894736842105
    - Precision, Recall, F1: (0.5875370919881305, 0.45622119815668205, 0.5136186770428015, None)
- 15 epochs: bidirectional model with attention:
    - Accuracy: 0.7526315789473684
    - Precision, Recall, F1: (0.5759162303664922, 0.5069124423963134, 0.5392156862745099, None)
- 10 epochs: bidirectional model with attention:
    - Accuracy: 0.75
    - Precision, Recall, F1: (0.5658536585365853, 0.5345622119815668, 0.5497630331753554, None)
- 5 epochs: bidirectional model with attention:
    - Accuracy: 0.7381578947368421
    - Precision, Recall, F1: (0.5481283422459893, 0.47235023041474655, 0.5074257425742573, None)

Also here's the W2V performance for those curious

**Word2Vec Embeddings 10-fold cross validation (30 epochs)**
- base model:
    - Accuracy: 0.7289473684210527
    - Precision, Recall, F1: (0.5273311897106109, 0.3822843822843823, 0.4432432432432432, None)
- bidirectional model:
    - Accuracy: 0.7263157894736842
    - Precision, Recall, F1: (0.5226480836236934, 0.34965034965034963, 0.41899441340782123, None)
- bidirectional model with attention:
    - Accuracy: 0.7157894736842105
    - Precision, Recall, F1: (0.49584487534626037, 0.4172494172494173, 0.4531645569620253, None)

**Word2Vec Embeddings 10-fold cross validation (40 epochs)**
- base model:
    - Accuracy: 0.7335526315789473
    - Precision, Recall, F1: (0.5298507462686567, 0.4965034965034965, 0.5126353790613718, None)
- bidirectional model:
    - Accuracy: 0.7342105263157894
    - Precision, Recall, F1: (0.5319693094629157, 0.48484848484848486, 0.5073170731707317, None)
- bidirectional model with attention:
    - Accuracy: 0.7217105263157895
    - Precision, Recall, F1: (0.5086206896551724, 0.4125874125874126, 0.4555984555984556, None)



# Other Notes

Note on using this repository:
- We've been experimenting with saving certain precomputed training/testing data tensors and loading them ahead of time so they don't have to be computed on the fly (saving quite a bit of setup!). However you'll need to download them from [here](https://drive.google.com/drive/folders/1Nr5rm54XH11B_55AJQylU6rMEPcGV4OO?usp=sharing) and place the "StoredTensors" folder in the root directory of your clone. They're simply too big to push/pull to/from github.


The original work was done by Michael Han, Anugrah Chemparathy, and Anson Hu as the final project for 6.864 (Advanced Natural Language Processing) in Fall 2021.

[training data](https://github.com/sjtuprog/fox-news-comments)
