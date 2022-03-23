# Implementation and Testing of Context-Aware LSTM Models for Hate Speech Detection
This repository contains code implementing and evalutating the robustness of architectures described in the paper [Detecting Online Hate Speech Using Context Aware Models](https://arxiv.org/pdf/1710.07395.pdf) as well as some extensions that we've been working on recently.

The purpose of these architectures is to take an internet comment and the title of the article it was written on and detect if the comment qualified as hate speech. The comments and titles are converted into vector embeddings with either Word2Vec (from Google) that are passed into parallel LSTM networks with attention. The concatenated outputs from the LSTM networks are then used to do a binary classification on whether the comment should be classified as hate speech or not.

This code was originally written for Google Colab to take advantage of GPU acceleration. It is currently being updated for clarity and some added functionality. Lately we've been working on using BERT to create embeddings (instead of Word2Vec) and fine tuning BERT for classification directly.

You can read our writeup on the original iteration of this project [here!](https://www.mit.edu/~anugrah/files/FinalProjectReport6864.pdf)

Note on using this repository:
- We've been experimenting with saving certain precomputed training/testing data tensors and loading them ahead of time so they don't have to be computed on the fly (saving quite a bit of setup!). However you'll need to download them from [here](https://drive.google.com/drive/folders/1Nr5rm54XH11B_55AJQylU6rMEPcGV4OO?usp=sharing) and place the "StoredTensors" folder in the root directory of your clone. They're simply too big to push/pull to/from github.


The original work was done by Michael Han, Anugrah Chemparathy, and Anson Hu as the final project for 6.864 (Advanced Natural Language Processing) in Fall 2021.

[training data](https://github.com/sjtuprog/fox-news-comments)
