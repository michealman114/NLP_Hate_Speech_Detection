import torch
import torch.nn as nn 
import torch.utils.data as torch_data

class CommentsDataset(torch.utils.data.Dataset):
    def __init__(self, comments, labels, titles = None):
        """
        comments/titles: (batch_size, max_length, embed_dim)
        labels: (batch_size,)
        """
        #Initialization
        self.data = comments
        self.titles = titles
        self.labels = labels
        self.length = self.labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select sample

        # Load data and get label
        X = self.data[index,:,:]
        y = self.labels[index]

        if self.titles is not None:
            t = self.titles[index,:,:]
            return X, t, y
        else:
            return X, y