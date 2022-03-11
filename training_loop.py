import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import gensim
import gensim.downloader as api


from datasets import *
from models import *
from process_data import *


def train(data, labels, n_epochs, batch_size, modeltype, model = None):
    device = torch.device('cuda')  # run on colab gpu

    if model is None:
        model = modeltype().to(device)
        
    opt = optim.Adam(model.parameters(), lr=0.001)

    training_data = Dataset(range(len(labels)), data, labels)

    loader = torch_data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.BCELoss()

    losses = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for context, label in loader:
            context = context.to(device)
            context = context.moveaxis(0, 1)
            label = label.to(device).type(torch.float32)

            preds = model.forward(context)

            opt.zero_grad()
            loss = loss_fn(preds, label)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        print('Loss:', epoch_loss)
        losses.append(epoch_loss)

    print(losses)
    return model


path = api.load("word2vec-google-news-300", return_path=True)
#print(path) #/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz

embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

train_lines = open("./Data/fox-news-comments.json", "r").readlines() #original 2015 data
test_lines = open("./Data/modern_comments.json", "r").readlines() #modern data

train_labels, train_comments, train_titles, train_max_len, train_max_title_len = clean(train_lines)
test_labels, test_comments, test_titles, test_max_len, test_max_title_len = clean(test_lines)

train_comment_array, train_title_array = to_array(embed, train_comments, train_titles, train_max_len, train_max_title_len)
test_comment_array, test_title_array = to_array(embed, test_comments, test_titles, test_max_len, test_max_title_len)

train_comment_array,train_title_array,train_labels = custom_shuffle(train_comment_array,train_title_array,train_labels)
test_comment_array, test_title_array, test_labels = custom_shuffle(test_comment_array, test_title_array, test_labels)

train_comment_array = np.float32(train_comment_array)
train_title_array = np.float32(train_title_array)

test_comment_array = np.float32(test_comment_array)
test_title_array = np.float32(test_title_array)




