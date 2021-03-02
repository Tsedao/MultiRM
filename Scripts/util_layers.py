import torch
from torch import nn
import pickle

class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights

class EmbeddingSeq(nn.Module):
    def __init__(self,weight_dict_path):
        """
        Inputs:
            weight_dict_path: path of pre-trained embeddings of RNA/dictionary
        """
        super(EmbeddingSeq,self).__init__()
        weight_dict = pickle.load(open(weight_dict_path,'rb'))

        weights = torch.FloatTensor(list(weight_dict.values())).cuda()
        num_embeddings = len(list(weight_dict.keys()))
        embedding_dim = 300

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = False

    def forward(self,x):

        out = self.embedding(x.type(torch.cuda.LongTensor))

        return out

class EmbeddingHmm(nn.Module):
    def __init__(self,t,out_dims):
        """
        Inputs:
            length: the length of input sequence
            t: the hyperparameters used for parallel message update iterations
            out_dims: dimension of new embedding
        """
        super(EmbeddingHmm,self).__init__()

        self.T = t
        self.out_dims = out_dims
        self.W1 = nn.Linear(4,out_dims)
        self.W2 = nn.Linear(out_dims,out_dims)
        self.W3 = nn.Linear(4,out_dims)
        self.W4 = nn.Linear(out_dims,out_dims)
        self.relu = nn.ReLU()

    def forward(self,x):
        """
        Inputs:
            x: RNA/DNA sequences using one-hot encoding, channel first: (bs,dims,seq_len)
        Outputs:
            miu: hmm encoding of RNA/DNA, channel last: (bs,seq_len,dims)
        """
        batch_size,length = x.shape[0], x.shape[-1]
        V = torch.zeros((batch_size,self.T+1,length+2,length+2,self.out_dims)).cuda()
        for i in range(1,self.T+1):
            for j in range(1,length+1):
                V[:,i,j,j+1,:] = self.relu(self.W1(x[:,:,j-1].clone())+self.W2(V[:,i-1,j-1,j,:].clone()))
                V[:,i,j,j-1,:] = self.relu(self.W1(x[:,:,j-1].clone())+self.W2(V[:,i-1,j+1,j,:].clone()))
        miu = torch.zeros((batch_size,length,self.out_dims)).cuda()

        for i in range(1,length+1):
            miu[:,i-1,:]= self.relu(self.W3(x[:,:,i-1].clone())+self.W4(V[:,self.T,i-1,i].clone())+self.W4(V[:,self.T,i+1,i].clone()))
        return miu

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_task):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_task = num_task
        self.log_vars = nn.Parameter(torch.zeros((num_task)))

    def forward(self,  y_pred,targets):

        def binary_cross_entropy(x, y):
            loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            return torch.sum(loss)
        # loss = nn.BCELoss(reduction='sum') fail to double backwards
        loss_output = 0
        for i in range(self.num_task):
            out = torch.exp(-self.log_vars[i])*binary_cross_entropy(y_pred[i],targets[:,i]) + self.log_vars[i]
            loss_output += out

        return loss_output
