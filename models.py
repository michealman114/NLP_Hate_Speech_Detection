import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(nn.Module): # single direction lstm, no attention
  def __init__(self, hidden_size = 100, embed_dim = 300):
    super(BaseModel, self).__init__()
    
    self.hidden_size = hidden_size
    
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = True, bidirectional = False)
    
    # two linear layers for context (final hidden state) => binary classification
    self.linear1 = nn.Linear(hidden_size, 150) 
    self.linear2 = nn.Linear(150, 1)

    self.relu = nn.ReLU()

    self.sigmoid = nn.Sigmoid()

  def forward(self, data):
    """
    data is an (N, L, D) = (batch_size, max_length, embed_dim) array
    returns an (N,1) array of binary probabilities that each comment is hateful
    """
    hidden_states, (_, _) = self.lstm(data) # (batch_size, max_length, hidden_size)
    
    sentences = torch.sum(hidden_states, axis = 1 ) # => (batch_size,hidden_size)

    return self.sigmoid(torch.squeeze(self.linear2(self.relu(self.linear1(sentences)))))

class Full_LSTM_Model(nn.Module):
  def __init__(self, hidden_size = 100, embed_dim = 300, bidi = True, attention = True):
    super(Full_LSTM_Model, self).__init__()
    if attention: assert bidi # attention only if the LSTM is bidirectional
    print(f"LSTM with bidirectional = {bidi}, attention = {attention}")
    
    self.hidden_size = hidden_size 
    self.attention = attention
    
    dropout = 0
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = 1, batch_first = True, dropout = dropout, bidirectional = bidi)

    # two linear layers for output of lstm: (final hidden state) => binary classification
    self.linear1 = nn.Linear(self.hidden_size*2 if bidi else self.hidden_size, 150) 
    self.linear2 = nn.Linear(150, 1)

    if self.attention: #assuming bidi
        self.attention1 = nn.Linear(2*hidden_size, 50) # map hidden state vector to value
        self.attention2 = nn.Linear(50, 1)
        self.sm = nn.Softmax(dim = 1)
    
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, data):
    """
    data is an (N, L, D) = (batch_size, max_length, embed_dim) array
    returns an (N,1) array of binary probabilities that each comment is hateful
    """
    hidden_states, (_, _) = self.lstm(data)
    # hidden_states = (batch_size, max_length, hidden_size) array
    
    """
    in this case, attention is a choice of coefficients which we use to weight hidden states 
    when summing them instead of adding them up with equal weighting
    
    TODO: add masks so that we aren't operating on all the hidden states since the padded ones don't matter!
    TODO: average sentences in non attention case instead of summing them
    """
    if self.attention:
        weights = self.attention1(hidden_states) #(batch_size,max_length,50)
        weights = self.relu(weights) #(batch_size,max_length,50)
        weights = self.attention2(weights) #(batch_size,max_length,1)
        alphas = self.sm(weights.squeeze()) #sm((batch_size,max_length)) => (batch_size,max_length)
        
        sentences = torch.sum(hidden_states * alphas[:,:,None], axis = 1) # (batch_size,hidden_size)
    
    else:
        sentences = torch.sum(hidden_states, axis = 1) # (batch_size, hidden_size)
    
    output = self.linear2(self.relu(self.linear1(sentences))) # => (batch_size,1)
    output = torch.squeeze(output) # => (batch_size)
    return self.sigmoid(output)

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
