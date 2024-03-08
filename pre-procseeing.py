#hide_output
from transformers import AutoTokenizer, BertTokenizer, BertModel
#from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from torch import nn
import torch
from math import sqrt 
from transformers import AutoConfig
import torch.nn.functional as F


model_ckpt = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)

text = "time flies like an arrow"
#show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
inputs.input_ids


config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
token_emb

inputs_embeds = token_emb(inputs.input_ids)
inputs_embeds.size()

query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()

weights = F.softmax(scores, dim=-1)
weights.sum(dim=-1)

attn_outputs = torch.bmm(weights, value)
attn_outputs.shape

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
    
multihead_attn = MultiHeadAttention(config)
print(inputs_embeds[0])
attn_output = multihead_attn(inputs_embeds)    
attn_output.size() 

#print(attn_output)