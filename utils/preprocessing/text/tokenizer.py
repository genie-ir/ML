import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


ger = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')
eng = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, val_data, test_data = Multi30k.splits(
    exts=['.de', '.en'],
    fields=[ger, eng]
)

ger.build_vocab(train_data, max_size=10000, min_freq=2)
eng.build_vocab(train_data, max_size=10000, min_freq=2)
