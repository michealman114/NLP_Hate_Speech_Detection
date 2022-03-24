import torch
import torch.nn as nn 
import torch.utils.data as torch_data

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, comments, labels, titles = None):
        """
        comments/titles: (batch_size, max_length, embed_dim)
        labels: (batch_size,)
        """
        #Initialization
        self.data = comments
        self.titles = titles
        self.labels = labels
        
        self.length = labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        comment = self.data[index]
        label = self.labels[index]

        if self.titles is not None:
            title = self.titles[index]
            return comment,title,label
        else:
            return comment,label

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, comments, comments_am, labels):
        """
        comments/titles: (batch_size, max_length, embed_dim)
        comments/titles_am: (batch_size,max_length) (the attention masks)
        labels: (batch_size,)
        """
        #Initialization
        self.comments = comments
        self.comments_am = comments_am

        self.labels = labels
        self.length = labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Load data and get label
        comment = self.comments[index]
        comment_am = self.comments_am[index]
        label = self.labels[index]

        return comment,comment_am,label