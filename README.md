# Implementation and Testing of Context-Aware LSTM Models for Hate Speech Detection
This repository contains code implementing and evalutating the robustness of architectures described in the paper [Detecting Online Hate Speech Using Context Aware Models](https://arxiv.org/pdf/1710.07395.pdf) as well as some extensions.

The comments and titles are converted with Word2Vec (from Google) to vectors that are passed into parallel LSTM networks with attention. The concatenated outputs from the LSTM networks are then used to do a binary classification on whether the comment should be classified as hate speech or not.

This code was originally written for Google Colab to take advantage of GPU acceleration. It is currently being updated for clarity and some added functionality.

[link to project write-up](https://www.mit.edu/~anugrah/files/FinalProjectReport6864.pdf)

The original work was done by Michael Han, Anugrah Chemparathy, and Anson Hu as the final project for 6.864 (Advanced Natural Language Processing) in Fall 2021.

[training data](https://github.com/sjtuprog/fox-news-comments)
