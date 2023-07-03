# import spacy
# from torchtext.datasets import Multi30k
# from torchtext.data import Field, BucketIterator
# import torchdata.datapipes as dp
# import torchtext.transforms as T
# from torchtext.vocab import build_vocab_from_iterator


from typing import Iterable, List
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    """
    Example:
        Tokenizer('de', 'en').get_vocabs()
    """
    def __init__(self, *langs, **build_vocab_params):
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3

        self.exts = []
        self.fields = []
        self.D = dict()
        self.vocabs = dict()

        # self.token_transform = {}
        self.vocab_transform = {}

        self.map_lang = { # python -m spacy download `lang-name`
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
        }

        for _lang in langs:
            lang = str(_lang).lower()
            print('!!!!!!!!!!!!!!!!', lang)
            # setattr(self, f'spacy_{lang}', spacy.load(self.map_lang.get(lang, lang)))
            setattr(self, f'spacy_{lang}', get_tokenizer('spacy', language=self.map_lang.get(lang, lang)))
            setattr(self, lang, Field(tokenize=self.tokenizer_lang(lang), lower=True, init_token='<sos>', eos_token='<eos>'))
            self.exts.append(f'.{lang}')
            self.fields.append(getattr(self, lang, None))

        self.langs = [ext.replace('.', '') for ext in self.exts]
        self.start(**build_vocab_params)

    def start(self, **build_vocab_params):
        """It can be overwrite in child class"""
        self.build_vocab(**build_vocab_params) # self.vocabs is ready to use, you shoud get vocabs with `.get_vocabs()` method.
    
    
    def yield_tokens(self, data_iter: Iterable, lang: str) -> List[str]:
        """helper function to yield list of tokens"""
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield getattr(self, f'spacy_{lang}')(data_sample[language_index[language]])

    def get_vocabs(self):
        return self.vocabs

    def tokenizer_lang(self, lang):
        spacy_lang = getattr(self, f'spacy_{lang}', None)
        assert spacy_lang is not None, '`spacy_lang=None` | for `lang={}`'.format(lang)
        def tokenizer_lang_slave(text):
            return [token.text for token in spacy_lang.tokenizer(text)]
        return tokenizer_lang_slave
    
    def __build_vocab(self, DiterList, DiterFunc):
        assert isinstance(getattr(self, 'build_vocab_params', None), dict)
        self.D = dict()
        self.vocabs = dict()
        for DiterKey in DiterList: # This is acording to keys defined in `Multi30k dataset` for other datasets this keys are be diffrents.
            self.vocabs[DiterKey] = dict()
            self.D[DiterKey] = DiterFunc(DiterKey=DiterKey)
            for lnng in self.langs:
                self.vocabs[DiterKey][lnng] = build_vocab_from_iterator(self.yield_tokens(self.D[DiterKey], lnng), specials=self.special_symbols, **self.build_vocab_params)
        
    def build_vocab(self, **build_vocab_params):
        """this function can be overwrite in child class"""
        build_vocab_params['min_freq'] = int(build_vocab_params.get('min_freq', 1))
        build_vocab_params['special_first'] = bool(build_vocab_params.get('special_first', True))
        # build_vocab_params['max_size'] = int(build_vocab_params.get('max_size', 10000))
        self.build_vocab_params = build_vocab_params
        self.build_vocab_params = 'mmd'
        DiterList = ['train', 'valid', 'test']
        def DiterFunc(DiterKey):
            return Multi30k(split=DiterKey, language_pair=self.langs)
        self.__build_vocab(DiterList, DiterFunc)