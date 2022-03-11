import torch
import torch.nn as nn 
import torch.utils.data as torch_data

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, data, labels, titles = None):
        # initialization now works with titles, passes in optional title information
        # works the same as before, but now gets title data if you give it to it
        'Initialization'
        self.data = data
        self.titles = titles
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.data[:,ID,:]
        y = self.labels[ID]

        if self.titles is not None:
          t = self.titles[:,ID,:]
          return X, t, y
        return X, y