from fastai.text.all import *

path = untar_data(URLs.IMDB)

files = get_text_files(path, folders=['train','test','unsup'])

txt = files[0].open().read()

spacy = WordTokenizer()
toks = first(spacy([txt]))

tkn = Tokenizer(spacy)

txts = L(o.open().read() for o in files[:2000])

def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])

toks = tkn(txt)
#print(coll_repr(tkn(txt),31))

toks200 = txts[:200].map(tkn)

num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20) 

nums = num(toks)[:20]

nums200 = toks200.map(num)

dl = LMDataLoader(nums200)

x,y = first(dl)

#print(x.shape, y.shape)

get_imdb = partial(get_text_files,folders=['train','test','unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

dls_lm.show_batch(max_n=2)

learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    metrics=[accuracy,Perplexity()]
).to_fp16()

learn.fit_one_cycle(1,2e-2)

learn.save('1epoch')

learn.save_encoder('finetuned')