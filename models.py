import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data

class BaseModel(nn.Module): # single direction lstm, no attention
  def __init__(self, hidden_size = 100, embed_dim = 300):
    super(BaseModel, self).__init__()
    
    self.hidden_size = hidden_size
    
    self.linear1 = nn.Linear(hidden_size, 150) # map context vector to value
    self.linear2 = nn.Linear(150, 1)

    self.relu = nn.ReLU()

    self.sigmoid = nn.Sigmoid()
    
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = False, dropout = 0.2, bidirectional = False)

  def forward(self, data):
    """
    data is an (L,N,D) array
    L = max_length of sequence
    N = batch_size
    D = embed_dim
    returns an (N,1) array of probabilities that each comment is hateful
    """
    hidden_states, (_, _) = self.lstm(data) # (L,N,H) array
    
    sentences = torch.sum(hidden_states, axis = 0)

    return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(sentences)))))

class BidiModel(nn.Module): # Bidi
  def __init__(self, hidden_size = 100, embed_dim = 300):
    super().__init__()
    
    self.hidden_size = hidden_size
    
    self.linear1 = nn.Linear(2*hidden_size, hidden_size) # map context vector to value
    self.linear2 = nn.Linear(hidden_size, 1)

    self.relu = nn.ReLU()

    self.sigmoid = nn.Sigmoid()
    
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = False, dropout = 0.2, bidirectional = True)

  def forward(self, data):
    """
    data is an (L,N,D) array
    L = max_length of sequence
    N = batch_size
    D = embed_dim
    returns an (N,1) array of probabilities that each comment is hateful
    """
    hidden_states, (_, _) = self.lstm(data) # (L,N,2H) array

    sentences = torch.sum(hidden_states, axis = 0)

    return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(sentences)))))


class BidiModel(nn.Module): # Bidi
  def __init__(self, hidden_size = 100, embed_dim = 300):
    super().__init__()
    
    self.hidden_size = hidden_size
    
    self.linear1 = nn.Linear(2*hidden_size, hidden_size) # map context vector to value
    self.linear2 = nn.Linear(hidden_size, 1)

    self.relu = nn.ReLU()

    self.sigmoid = nn.Sigmoid()
    
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = False, dropout = 0.2, bidirectional = True)

  def forward(self, data):
    """
    data is an (L,N,D) array
    L = max_length of sequence
    N = batch_size
    D = embed_dim
    returns an (N,1) array of probabilities that each comment is hateful
    """
    hidden_states, (_, _) = self.lstm(data) # (L,N,2H) array

    sentences = torch.sum(hidden_states, axis = 0)

    return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(sentences)))))