import torch
import torch.nn as nn 
import torch.utils.data as torch_data

class CommentsDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, comments, labels, titles = None):
        """
        comments/titles: (batch_size, max_length, embed_dim)
        labels: (batch_size,)
        """
        #Initialization
        self.data = comments
        self.titles = titles
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.data[ID,:,:]
        y = self.labels[ID]

        if self.titles is not None:
            t = self.titles[ID,:,:]
            return X, t, y
        else:
            return X, y