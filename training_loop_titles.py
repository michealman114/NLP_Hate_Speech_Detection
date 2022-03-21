import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import gensim
import gensim.downloader as api


from datasets import *
from models import *
from data_processing import *



def train_with_titles(data, titles, labels, n_epochs, batch_size, model = None):
    device = torch.device('cuda')  # run on colab gpu

    if model is None:
        model = ModelWithTitle().to(device)
        
    opt = optim.Adam(model.parameters(), lr=0.001)

    training_data = CommentsDataset(range(len(labels)), data, labels, titles = titles)

    loader = torch_data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.BCELoss()

    losses = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for context, t, label in loader:
            context = context.to(device)
            context = context.moveaxis(0, 1)
            t = t.to(device)
            t = t.moveaxis(0,1)
            label = label.to(device).type(torch.float32)

            preds = model.forward(context, t)

            opt.zero_grad()
            loss = loss_fn(preds, label)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        print('Loss:', epoch_loss)
        losses.append(epoch_loss)

    print(losses)
    return model




