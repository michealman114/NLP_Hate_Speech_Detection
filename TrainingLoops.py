import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data

import sklearn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from CommentsDatasets import *
from tqdm import tqdm

def train(training_dataset, n_epochs, batch_size, device, modeltype, model = None, embed_dim = None, bidi = False, attention = False, check_data = None):
    intermediate_checks = True if (check_data is not None) else False

    if model is None:
        """
        embed_dim = 768 for BERT embeddings
        embed_dim = 300 for Word2Vec embeddings 
        """
        model = modeltype(embed_dim = embed_dim, bidi=bidi, attention=attention).to(device)
    else:
        model = model.to(device)
        
    opt = optim.Adam(model.parameters(), lr=0.001)

    loader = torch_data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.BCELoss()

    losses = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for comment, comment_pm, label in loader:
            comment = comment.to(device)
            label = label.to(device).type(torch.float32)

            preds = model.forward(comment)

            opt.zero_grad()
            loss = loss_fn(preds, label)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        print('Loss:', epoch_loss)
        losses.append(epoch_loss)

        if intermediate_checks and (epoch % 5 == 0):
            print(f'Intermediate Performance Evaluation for Epoch {epoch}')
            test_comments, test_comments_pm, test_labels = check_data            
            iter_loss, (y_pred, y_true) = test_model(model, test_comments, test_comments_pm, test_labels, device)
            y_pred = torch.round(y_pred).cpu().detach().numpy()
            print('Accuracy:',accuracy_score(y_true,y_pred))
            print('Precision, Recall, F1:',precision_recall_fscore_support(y_true, y_pred, average='binary'))
            model.train()
    
    return model,losses


def kfold_crossvalidation(k, comments, comments_pm, labels, modeltype, device, n_epochs = 30, embed_dim = 300, bidi = False, attention = False, batch_size = 128, intermediate_checks = False):
    
    num_samples , _ , _ = comments.shape
    fraction = 1/k
    seg = int(num_samples * fraction)
    segment_indices = []
    for i in range(k):
        vall = i * seg
        valr = i * seg + seg
        segment_indices.append(list(range(vall,valr)))
    
    all_preds = []
    all_labels = []
    # run the ith split
    for i in range(k):
        print('\n=======================================\n')
        print(f'running split {i+1}:\n')
        train_indices = []
        test_indices = segment_indices[i]
        for j in range(k):
            if j != i:
                train_indices.extend(segment_indices[j])

    
        train_comments_i = comments[train_indices,:,:]
        train_comments_pm_i = comments_pm[train_indices,:]
        train_labels_i = labels[train_indices]

        test_comments_i = comments[test_indices,:,:]
        test_comments_pm_i = comments_pm[test_indices,:]
        test_labels_i = labels[test_indices]

        training_data = GeneralDataset(train_comments_i, train_comments_pm_i, train_labels_i)
        #def train(training_dataset, n_epochs, batch_size, device, modeltype, model = None, embed_dim = None, bidi = False, attention = False):
        if intermediate_checks:
            check_data = [test_comments_i, test_comments_pm_i, test_labels_i]
            model_i, loss_i = train(training_data, n_epochs, batch_size, device, modeltype, embed_dim = embed_dim, bidi = bidi, attention = attention, check_data = check_data)
        else:
            model_i, loss_i = train(training_data, n_epochs, batch_size, device, modeltype, embed_dim = embed_dim, bidi = bidi, attention = attention)

        iter_loss, (y_pred, y_true) = test_model(model_i, test_comments_i, test_comments_pm_i, test_labels_i, device)
        y_pred = torch.round(y_pred).cpu().detach().numpy()
        all_preds.append(y_pred)
        all_labels.append(y_true)
        print(f'Final Evaluation for model {i+1}')
        print('Accuracy:',accuracy_score(y_true,y_pred))
        print('Precision, Recall, F1:',precision_recall_fscore_support(y_true, y_pred, average='binary'))
    
    print('\n===Aggregate Stats===')
    p = np.concatenate(all_preds, axis = None)
    l = np.concatenate(all_labels, axis = None)
    print('Accuracy:', accuracy_score(l, p))
    print('Precision, Recall, F1:', precision_recall_fscore_support(l, p, average = 'binary'))

def test_model(model, test_comments, test_comments_pm, test_labels, device):
    test_dataset = GeneralDataset(test_comments, test_comments_pm, test_labels)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=len(test_labels))
    
    loss_fn = nn.BCELoss()
    predictions = None

    model.eval()
    with torch.no_grad():
        for comment, comment_pm, label in test_loader:
            comment = comment.to(device)
            label = label.to(device).type(torch.float32)

            preds = model.forward(comment)
            predictions = preds

            loss = loss_fn(preds, label)
            print(loss.item())
    
    return loss.item(), (preds, test_labels)