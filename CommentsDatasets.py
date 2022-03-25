import torch
import torch.nn as nn 
import torch.utils.data as torch_data


class GeneralDataset(torch.utils.data.Dataset):
    """"
    Dataset that works with attention masks
    """
    def __init__(self, comments, comments_pm, labels):
        """
        comments/titles: (batch_size, max_length, embed_dim)
        #comments/titles_pm: (batch_size,max_length) (the padding masks)
        labels: (batch_size,)
        """
        #Initialization
        self.comments = comments
        self.comments_pm = comments_pm

        self.labels = labels
        self.length = labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        comment = self.comments[index]
        comment_pm = self.comments_pm[index]
        label = self.labels[index]

        return comment,comment_pm,label