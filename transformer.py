import torch
import torch.nn as nn
import numpy as np

from fastai.text.all import *

path = untar_data(URLs.HUMAN_NUMBERS)

lines=L()
with open(path/'train.txt') as f: lines+=L(*f.readlines())
with open(path/'valid.txt') as f: lines+=L(*f.readlines())

text = ' . '.join([l.strip() for l in lines])

tokens = text.split(' ')

vocab = L(*tokens).unique()

word2idx = {w:i for i,w in enumerate(vocab)}
nums = L(word2idx[i] for i in tokens)

def group_chunks(ds, bs):
    m = len(ds) // bs
    new_ds = L()
    for i in range(m): new_ds += L(ds[i + m*j] for j in range(bs))
    return new_ds

bs = 64
sl = 16
#seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1])) for i in range(0,len(nums)-sl-1,sl))
seqs = L(tensor(nums[i:i+sl], dtype=torch.float32) for i in range(0,len(nums)-sl-1,sl))

cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)

"""class LMModel4(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = 0
    
    def forward(self, x):
        outs = []
        #print(x)
        for i in range(sl):
            self.h = self.h + self.i_h(x[:,i])
            self.h = F.relu(self.h_h(self.h))
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1)
    
    def reset(self): self.h = 0
    
def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))

learn = Learner(dls,LMModel4(len(vocab), 64), loss_func=loss_func,
                metrics=accuracy, cbs=ModelResetter)

learn.fit_one_cycle(5, 3e-3)"""

# Attention mechanism 
# attention_weight = softmax(QK/sqrt(dk))
# output = attention_weight * Value

def attention_score(q, k, v):
    dim = q.size(-1)
    print(q)
    print(q.shape)
    print(k.transpose(1,2))
    score = torch.bmm(q, k.transpose(0,1)) / torch.sqrt(dim)
    attention_weights = F.softmax(score, dim=-1)
    return torch.bmm(attention_weights,v)


class AttentionHead(Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, hidden_state):
        attention = attention_score(self.query(hidden_state), self.key(hidden_state), self.value(hidden_state))
        return attention 

    def reset(self): pass
    
def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))

attention = AttentionHead(len(seqs[0]), len(seqs[0]))
value = attention(seqs[0])

print(value)

learn = Learner(dls,AttentionHead(len(vocab), 64), loss_func=loss_func,
                metrics=accuracy, cbs=ModelResetter)

learn.fit_one_cycle(5, 3e-3)

"""q = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)
k = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)
v = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)

attention = AttentionHead(3,3)
value = attention(q,k,v)

print(value)
"""
 
"""class AttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = torch.nn.Linear(input_dim, hidden_dim)
        self.key = torch.nn.Linear(input_dim, hidden_dim)
        self.value = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, q, k, v):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        
        k_transposed = key.transpose(0,1)  
        attention_value = torch.softmax(query.mm(k_transposed)/torch.sqrt(torch.tensor(query.shape[1], dtype=torch.float32)), dim=1)
        
        return attention_value * value  
        
    
q = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)
k = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)
v = torch.tensor([1,2,3],dtype=torch.float32).unsqueeze(0)

attention = AttentionHead(3,3)
value = attention(q,k,v) 

print(value)"""



  
