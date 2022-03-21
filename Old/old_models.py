"""
Deprecated models
- These use batch_first = False
- Not as functional and lots of repetition
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(nn.Module):
    """
    Vanilla LSTM, no attention
    """
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

class BidiModel(nn.Module):
    """
    Bidirectional LSTM (no attention)
    """
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

class FullModel(nn.Module): 
    """
    Bidirectional LSTM with attention
    """
    def __init__(self, hidden_size = 100, embed_dim = 300):
        super().__init__()
        
        self.hidden_size = hidden_size
        #self.embedding = embed
        
        self.linear1 = nn.Linear(2*hidden_size, hidden_size) # map context vector to value
        self.linear2 = nn.Linear(hidden_size, 1)

        self.attention1 = nn.Linear(2*hidden_size, 50) # map hidden state vector to value
        self.attention2 = nn.Linear(50, 1)

        self.relu = nn.ReLU()

        self.sm = nn.Softmax(dim = 0)
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
        weights = self.attention2(self.relu(self.attention1(hidden_states))) # (L,N,1) array
        
        alpha = self.sm(weights.reshape(weights.shape[:-1])) # weights

        hidden_states = torch.moveaxis(hidden_states, -1, 0)


        sentences = torch.sum(hidden_states * alpha, axis = 1)

        sentences = torch.moveaxis(sentences, 0, -1)

        return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(sentences)))))

class ModelWithTitle(nn.Module):
    """
    Bidrectional LSTM with attention
    Augmented with title data
    """
    def __init__(self, hidden_size = 100, embed_dim = 300):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(4*hidden_size, hidden_size) # map context vector to value (concatenated from parallel networks)
        self.linear2 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

        self.sm = nn.Softmax(dim = 0)
        self.sigmoid = nn.Sigmoid()

        #comments
        self.attention1_comment = nn.Linear(2*hidden_size, 50) # map hidden state vector to value
        self.attention2_comment = nn.Linear(50, 1)
        
        self.lstm_comment = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = False, dropout = 0.2, bidirectional = True)

        #titles
        self.attention1_title = nn.Linear(2*hidden_size, 50)
        self.attention2_title = nn.Linear(50, 1)

        self.lstm_title = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = False, dropout = 0.2, bidirectional = True)

    def forward(self, comment_data, title_data):
        """
        comments is an (L1,N,D) array
        titles is an (L2,N,D) array
        L1 = max_length of sequence
        L2 = max_length of title
        N = batch_size
        D = embed_dim
        returns an (N,1) array of probabilities that each comment is hateful
        """
        hidden_states, (_, _) = self.lstm_comment(comment_data) # (L,N,2H) array

        weights = self.attention2_comment(self.relu(self.attention1_comment(hidden_states))) # (L,N,1) array
        
        alpha = self.sm(weights.reshape(weights.shape[:-1])) # weights

        hidden_states = torch.moveaxis(hidden_states, -1, 0)

        sentences = torch.sum(hidden_states * alpha, axis = 1)

        sentences = torch.moveaxis(sentences, 0, -1) # sentences is N x 2*hidden_size

        hidden_states, (_,_) = self.lstm_title(title_data)
        weights = self.attention2_title(self.relu(self.attention1_title(hidden_states)))
        alpha = self.sm(weights.reshape(weights.shape[:-1]))
        hidden_states = torch.moveaxis(hidden_states, -1, 0)
        titles = torch.sum(hidden_states * alpha, axis = 1)
        titles = torch.moveaxis(titles, 0, -1) # titles is N x 2*hidden_size

        result = torch.cat((sentences, titles), dim = 1)

        return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(result)))))
