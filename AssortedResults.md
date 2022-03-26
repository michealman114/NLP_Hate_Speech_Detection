# Performance of BERT fine-tuned
In most modern research Transformer models substantially outperform sequence based models, so we decided to also try fine-tuning variants of BERT-base, which proved to be quite an involved process for which we reconstructed most of our infrastructure (see ClassificationWithBERT.ipynb for some of our training setups).

We set up 3 different BERT models - BERTForSequenceClassification,DistilBERTForSequenceClassification, and a customized model built on top of a base DistilBERTModel. We can take advantage of BERT's next sentence prediction training history (which stores information needed for classification in the first element of the encoded output) to fine-tune the model. The exact setup of this is most relevant for our customized model which you can see in the colab notebook listed above.


Here are some comments about training in general:
- Initially we had an issue where the model would acheive good performance by always reporting 'not hate speech' because the dataset was originally unbalanced. We tried to counter this by rebalancing the dataset by randomly clipping negative-labeled comments until our dataset was balanced.
- However this led to issues with the hyperparameters which mattered enormously (a lesson learned the hard way). Training on too high of a learning rate didn't really work because the model would just oscillate between classifying everything as negative, and classifying everything as positive. We ended up trying a range of learning rates (between 1e-5 and 5e-4) before finally settling on 2e-5 (which was on the lower end of values suggested in the paper). Also setting up (a simple) LR scheduler empirically worked very well here.
- You really don't need many epochs to fine-tune BERT, even just 3 

*Fine-tuning BERTForSequenceClassification:*

We trained BERTForSequenceClassification with 700 samples and tested on the remaining 170.

On just 2 epochs, it had achieved extremely good results, substantially outperforming the author's results:
- Accuracy: 0.7848837209302325
- Precision, Recall, F1: (0.7261904761904762, 0.8133333333333334, 0.7672955974842767, None)

The performance with 3 epochs was not quite as good (overfitting), but the performance with 2 epochs is enough to be extremely impressive.
- Accuracy: 0.7267441860465116
- Precision, Recall, F1: (0.6346153846153846, 0.88, 0.7374301675977655, None)

*Fine-tuning DistilBERT:*

When we finally tuned the parameters and set up training perfectly we acheived extremely good performance from DistilBERT:
DistilBERT for sequence classification results (fine-tuned on 100, tested on 50):
- Accuracy: 0.91
- Precision, Recall, F1: (0.8656716417910447, 1.0, 0.928, None)

However this performance was on a small test dataset of 50 samples (never before seen, but nonetheless a small sample size), so its hard to say in retrospect if it would have performed this well on the whole. 

From manual testing with custom sentences however this model worked pretty well and intuitively. I didn't save this one unfortunately since I tried reproducing it immediately to no success.

Here is our DistilBERT variant's performance after being fine-tuned on progressively less and less data.  Performance holds even with just 300 training samples, and substantially outperforms the authors in terms of viability (we have substantially better better precision/recall/F1) although this is not surprising since Transformers are so much more successful in general.

DistilBERT for sequence classification results (fine-tuned on 700, validated on 170)
- Test Performance
    - Accuracy: 0.7325581395348837
    - Precision, Recall, F1: (0.6746987951807228, 0.7466666666666667, 0.708860759493671, None)


DistilBERT for sequence classification results (fine-tuned on 300, validated on 100, tested on remaining 470)
- Validation Performance
    - Accuracy: 0.71
    - Precision, Recall, F1: (0.7254901960784313, 0.7115384615384616, 0.7184466019417477, None)
- Test Performance
    - Accuracy: 0.65
    - Precision, Recall, F1: (0.6115702479338843, 0.6883720930232559, 0.6477024070021883, None)
- 3 epochs were run again with model.eval() mode, validation results:
    - Accuracy: 0.75
    - Precision, Recall, F1: (0.7454545454545455, 0.7884615384615384, 0.766355140186916, None)
- 3 more epochs Test:
    - Accuracy: 0.717391304347826
    - Precision, Recall, F1: (0.683982683982684, 0.7348837209302326, 0.7085201793721975, None)


DistilBERT for sequence classification results (fine-tuned on 100, validated on 50, tested on remaining 720)
- Validation Performance
    - Accuracy: 0.76
    - Precision, Recall, F1: (0.717948717948718, 0.9655172413793104, 0.8235294117647058, None)
- Test Performance
    - Accuracy: 0.6366197183098592
    - Precision, Recall, F1: (0.5748175182481752, 0.9264705882352942, 0.7094594594594595, None)

This model was very overzealous with classifying things as hate speech (low precision), but it also almost never missed hate speech (high recall).


# Performance of LSTMs on pre-trained embeddings

Here we can compare the performance of our models with those of the author.  First up, the performance of BERT embeddings is quite good. Even without much optimization, the performance below is on par with the author's best results:

**BERT Embeddings 10-fold cross validation (not augmented with title)**
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

Training on BERT embeddings works best with 10 epochs (15 epochs isn't bad either, but 10 just slightly outperforms it). Yay for no more overfitting.

Here's the W2V performance for comparison:

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

Training any further yields no substantial improvement in performance:

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


# Miscellaneous
For those curious, here's how stuff works if you try training the non-attention LSTM models with BERT embeddings:

**BERT Embeddings Results (from a lazy training loop on 30 epochs)**
- base model: Loss: 858.4895858764648
- bidirectional LSTM: Loss: 341.25000190734863
- bidirectional LSTM with Attention: Loss: 0.0014144671586109325

These results are both intuitive and surprising. It is amsuingly surprising that the LSTMS without attention are as ludicrously terrible as they are, but it kind of makes sense.  The enormous performance jump suggests that the attention mechanism (which in this application is just a very simple set of FC layers) is doing most of the work (even without being attention masked - which is something we need to fix pretty urgently). I'd be willing to bet that when working on BERT embeddings we can just trivially slap on a couple linear layers on top and get really good performance.

Also, when we were cleaning up the dataset before feeding into word2vec we originally did some classical stuff (removing stopwords, punctuation etc) that removes important embedding context - especailly since some stopwords like "no","never","not" substantially change the meaning of a sentence - which probably also explains a good amount of why BERT performs so much better. Fixing this really obvious data processing mistake isn't too difficult, but we can do that later.



