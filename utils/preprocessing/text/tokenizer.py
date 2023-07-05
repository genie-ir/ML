# https://spacy.io/
# https://www.nltk.org/

import torch
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from libs.basicDS import def_instance_method
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator
from utils.pl.tools.dataset import DataModuleFromConfig

class Tokenizer:
    """
    Example:
        Tokenizer('de', 'en').get_vocabs()
    """
    def __init__(self, *langs, **build_vocab_params):
        
        self.map_lang = { # python -m spacy download en_core_web_sm
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
        }
        
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        for iss, ss in enumerate(self.special_symbols):
            setattr(self, ss.replace('<', '').replace('>', '').upper() + '_IDX', iss)
        
        self.langs = []
        for _lang in langs:
            lang = str(_lang).lower()
            self.langs.append(lang)
            setattr(self, f'spacy_{lang}', get_tokenizer('spacy', language=self.map_lang.get(lang, lang)))
        self.idxlangs = dict((lang, idx) for idx, lang in enumerate(self.langs))

        self.build_vocab(**build_vocab_params)

    @property
    def dataloaders(self):
        return self.__dataloaders
    
    @property
    def len(self):
        return self.__vocabs_len

    def get_mapping(self, DiterKey, lang):
        return self.__vocabs[DiterKey][lang].get_stoi()

    def sequential_transforms(self, *transforms):
        """helper function to club together sequential operations"""
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
    
    def tensor_transform(self, token_ids: List[int]):
        """function to add BOS/EOS and create tensor for input sequence indices"""
        return torch.tensor([self.BOS_IDX] + token_ids + [self.EOS_IDX])

    def yield_tokens(self, data_iter: Iterable, lang: str) -> List[str]:
        """helper function to yield list of tokens. this used once to generate a mapping from each unique string_token to coresponding int number"""
        for data_sample in data_iter: # raw data iterator with help of yield technic
            yield getattr(self, f'spacy_{lang}')(data_sample[self.idxlangs[lang]])

    def collate_fn(self, batch, **kwargs):
        """function to collate data samples into batch tensors"""
        out = [[] for lang in self.langs] # output batch is a tensor version of input batch
        for B in batch:
            for idx, b in enumerate(B):
                out[idx].append(self.__text_transform[kwargs['memory']['DiterKey']][self.langs[idx]](b.rstrip('\n')))
        for idx_outi, outi in enumerate(out):
            out[idx_outi] = pad_sequence(outi, padding_value=self.PAD_IDX).transpose(0, 1)
        return out

    def __build_vocab(self, DiterList, DiterFunc):
        assert isinstance(getattr(self, 'build_vocab_params', None), dict)
        self.__dataloaders =  dict()

        self.__D = dict()
        self.__vocabs = dict()
        self.__vocabs_len = dict()
        self.__text_transform = dict()
        
        for DiterKey in DiterList:
            self.__vocabs[DiterKey] = dict()
            self.__vocabs_len[DiterKey] = dict()
            self.__text_transform[DiterKey] = dict()
            self.__D[DiterKey] = DiterFunc(DiterKey=DiterKey)

            for lang in self.langs:
                self.__vocabs[DiterKey][lang] = build_vocab_from_iterator(self.yield_tokens(self.__D[DiterKey], lang), specials=self.special_symbols, **self.build_vocab_params)
                self.__vocabs[DiterKey][lang].set_default_index(self.UNK_IDX)
                self.__vocabs_len[DiterKey][lang] = len(self.__vocabs[DiterKey][lang])
                self.__text_transform[DiterKey][lang] = self.sequential_transforms(
                    getattr(self, f'spacy_{lang}', None), #Tokenization
                    self.__vocabs[DiterKey][lang], #Numericalization -> `build_vocab_from_iterator`
                    self.tensor_transform # Add BOS/EOS and create tensor
                )
            self.__dataloaders[DiterKey] = getattr(DataModuleFromConfig(
                **{
                    DiterKey: {
                        'params': {'dataset': self.__D[DiterKey]}
                    }
                },
                batch_size=3,
                use_dck_mapper=False,
                dataset_category=[DiterKey],
                custom_collate=def_instance_method(self, f'_{DiterKey}_collate_fn', self.collate_fn, DiterKey=DiterKey)
            ), f'_{DiterKey}_dataloader')()
    
    def build_vocab(self, **build_vocab_params):
        """this function can be overwrite in child class"""
        build_vocab_params['min_freq'] = int(build_vocab_params.get('min_freq', 1))
        build_vocab_params['special_first'] = bool(build_vocab_params.get('special_first', True))
        self.build_vocab_params = build_vocab_params
        DiterList = ['train', 'valid', 'test']
        def DiterFunc(DiterKey):
            return Multi30k(split=DiterKey, language_pair=self.langs)
        self.__build_vocab(DiterList, DiterFunc)