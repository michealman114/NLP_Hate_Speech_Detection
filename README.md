# Implementation and Testing of Context-Aware LSTM Models for Hate Speech Detection
Implementing and testing the architecture described in the paper [Detecting Online Hate Speech Using Context Aware Models](https://arxiv.org/pdf/1710.07395.pdf).

The comments and titles are converted with Word2Vec (from Google) to vectors that are passed into parallel LSTM networks with attention. The concatenated outputs from the LSTM networks are then used to do a binary classification on whether the comment should be classified as hate speech or not.

This code was originally written for Google Colab. This is a reproduction of the code for extra organization and clarity.

[link to project write-up](https://www.mit.edu/~anugrah/files/FinalProjectReport6864.pdf)

This was done by Michael Han, Anugrah Chemparathy, and Anson Hu as the final project for 6.864 (Advanced Natural Language Processing) in Fall 2021.

[training data](https://github.com/sjtuprog/fox-news-comments)