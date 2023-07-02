import spacy
from torchtext.datasets import Multi30k
# from torchtext.data import Field, BucketIterator
import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    """
    Example:
        Tokenizer('de', 'en').get_vocabs()
    """
    def __init__(self, *langs, **build_vocab_params):
        self.exts = []
        self.fields = []
        self.D = dict()
        self.vocabs = dict()

        self.map_lang = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
        }

        for _lang in langs:
            lang = str(_lang).lower()
            print('!!!!!!!!!!!!!!!!', lang)
            setattr(self, f'spacy_{lang}', spacy.load(self.map_lang.get(lang, lang)))
            setattr(self, lang, Field(tokenize=self.tokenizer_lang(lang), lower=True, init_token='<sos>', eos_token='<eos>'))
            self.exts.append(f'.{lang}')
            self.fields.append(getattr(self, lang, None))

        self.start(**build_vocab_params)

    def start(self, **build_vocab_params):
        """It can be overwrite in child class"""
        self.build_vocab(**build_vocab_params) # self.vocabs is ready to use, you shoud get vocabs with `.get_vocabs()` method.
    
    def get_vocabs(self):
        return self.vocabs

    def tokenizer_lang(self, lang):
        spacy_lang = getattr(self, f'spacy_{lang}', None)
        assert spacy_lang is not None, '`spacy_lang=None` | for `lang={}`'.format(lang)
        def tokenizer_lang_slave(text):
            return [token.text for token in spacy_lang.tokenizer(text)]
        return tokenizer_lang_slave
    
    def build_vocab(self, **build_vocab_params):
        """this function can be overwrite in child class"""
        build_vocab_params['min_freq'] = int(build_vocab_params.get('min_freq', 2))
        build_vocab_params['max_size'] = int(build_vocab_params.get('max_size', 10000))
        self.build_vocab_params = build_vocab_params

        self.D = dict()
        self.D['train'], self.D['val'], self.D['test'] = Multi30k.splits(
            exts=self.exts,
            fields=self.fields
        )
        
        self.vocabs = dict()
        for dataset_key in list(self.D.keys()):
            self.vocabs[dataset_key] = dict()
            for idx, ext in enumerate(self.exts):
                self.vocabs[dataset_key][ext.replace('.', '')] = self.fields[idx].build_vocab(self.D[dataset_key], **self.build_vocab_params)


