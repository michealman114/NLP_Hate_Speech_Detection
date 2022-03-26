# Implementation and Evaluation of Deep Context-Aware Models for Hate Speech Detection
This repository contains code implementing and evalutating the robustness of architectures described in the paper [Detecting Online Hate Speech Using Context Aware Models](https://arxiv.org/pdf/1710.07395.pdf) as well as some extensions that we've been working on recently.

The purpose of these architectures is to take an internet comment and the title of the article it was written on and detect if the comment qualified as hate speech. The comments and titles are converted into vector embeddings with either Word2Vec (from Google) that are passed into parallel LSTM networks with attention. The concatenated outputs from the LSTM networks are then used to do a binary classification on whether the comment should be classified as hate speech or not.

This code was originally written for Google Colab to take advantage of GPU acceleration. It is currently being updated for clarity and some added functionality. Here are two of the big extensions we've added:
- Finetuning different versions of BERT (BERTForSequenceClassification,DistilBERTForSequenceClassification, and a customized model built on top of a base DistilBERTModel) for classification performance, which outperforms all of the authors listed models without even requiring title context.
- use of BERT embeddings instead of Word2Vec embeddings to get better left right contextualization for use in a bidirectional LSTM.
- Substantially better infrastructure for processing, editing, and storing embeddings ahead of time to significantly streamline training.



You can read our writeup on the original iteration of this project [here](https://www.mit.edu/~anugrah/files/FinalProjectReport6864.pdf)


# Recent Results
Some new results - we evaluated the performance of variants of BERT for hate speech classification
DistilBERT for sequence classification results (fine-tuned on 700, Tested on 170) - 3 epochs
- Test Performance
    - Accuracy: 0.7325581395348837
    - Precision, Recall, F1: (0.6746987951807228, 0.7466666666666667, 0.708860759493671)

DistilBERT for sequence classification results (fine-tuned on 300, validated on 100, tested on remaining 470)
- Test Performance - 3 epochs
    - Accuracy: 0.65
    - Precision, Recall, F1: (0.6115702479338843, 0.6883720930232559, 0.6477024070021883, None)
- Test Performance - 3 more epochs of training
    - Accuracy: 0.717391304347826
    - Precision, Recall, F1: (0.683982683982684, 0.7348837209302326, 0.7085201793721975, None)

You can see more of our results from this new iteration of the project in the file "AssortedResults.md"


# Other Notes

Note on using this repository:
- We've been experimenting with saving certain precomputed training/testing data tensors and loading them ahead of time so they don't have to be computed on the fly (saving quite a bit of setup!). However you'll need to download them from [here](https://drive.google.com/drive/folders/1Nr5rm54XH11B_55AJQylU6rMEPcGV4OO?usp=sharing) and place the "StoredTensors" folder in the root directory of your clone. They're simply too big to push/pull to/from github.


The original work was done by Michael Han, Anugrah Chemparathy, and Anson Hu as the final project for 6.864 (Advanced Natural Language Processing) in Fall 2021.
