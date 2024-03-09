import torch
import torch.nn as nn
import numpy as np

from fastai.text.all import *

path = untar_data(URLs.IMDB)
files = get_text_files(path, folders=['train','test','unsup'])

#print(len(files))

txt = files[0].open().read()

spacy = WordTokenizer()
toks = first(spacy([txt]))

tkn = Tokenizer(spacy)

txts = L(o.open().read() for o in files[:2])

print(txts[0])

toks = tkn(txt)

toks200 = txts[:2].map(tkn)

num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20) 

nums = num(toks)[:20]

nums200 = toks200.map(num)

print(nums200[0])

dl = LMDataLoader(nums200)

x,y = first(dl)

print(x.shape, y.shape)

get_imdb = partial(get_text_files,folders=['train','test','unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

dls_lm.show_batch(max_n=2)
print(dls_lm)

"""for data in dls_lm:
    print(data)
    break"""

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
    
attention = AttentionHead(72,72)
pred = attention(nums200[0])

learn = language_model_learner(
    dls_lm, attention, drop_mult=0.3,
    metrics=[accuracy,Perplexity()]
).to_fp16()

