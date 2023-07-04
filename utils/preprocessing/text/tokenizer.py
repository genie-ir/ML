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
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3

        self.langs = []
        for _lang in langs:
            lang = str(_lang).lower()
            self.langs.append(lang)
            setattr(self, f'spacy_{lang}', get_tokenizer('spacy', language=self.map_lang.get(lang, lang)))
        self.idxlangs = dict((lang, idx) for idx, lang in enumerate(self.langs))

        self.build_vocab(**build_vocab_params)

    def yield_tokens(self, data_iter: Iterable, lang: str) -> List[str]:
        """helper function to yield list of tokens"""
        for data_sample in data_iter: # raw data iterator with help of yield technic
            yield getattr(self, f'spacy_{lang}')(data_sample[self.idxlangs[lang]])

    @property
    def dataloaders(self):
        return self.__dataloaders

    def sequential_transforms(self, *transforms):
        """helper function to club together sequential operations"""
        def func(txt_input):
            for transform in transforms:
                print(txt_input)
                print(transform)
                txt_input = transform(txt_input)
                print(txt_input)
                print('-'*30)
            return txt_input
        return func
    
    def tensor_transform(self, token_ids: List[int]):
        """function to add BOS/EOS and create tensor for input sequence indices"""
        return torch.cat((torch.tensor([self.BOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])))

    def collate_fn(self, batch, **kwargs):
        """function to collate data samples into batch tensors"""
        print('@@@@@@@', batch)
        out = [[] for lang in self.langs] # output batch is a tensor version of input batch
        for B in batch:
            for idx, b in enumerate(B):
                out[idx].append(self.__text_transform[kwargs['memory']['DiterKey']][self.langs[idx]](b.rstrip('\n')))
        for idx_outi, outi in enumerate(out):
            out[idx_outi] = pad_sequence(outi, padding_value=self.PAD_IDX)
        assert False
        return out

        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.__text_transform[SRC_LANGUAGE](src_sample.rstrip('\n')))
            tgt_batch.append(self.__text_transform[TGT_LANGUAGE](tgt_sample.rstrip('\n')))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch
    
    def __build_vocab(self, DiterList, DiterFunc):
        assert isinstance(getattr(self, 'build_vocab_params', None), dict)
        self.__dataloaders =  dict()

        self.__D = dict()
        self.__vocabs = dict()
        self.__text_transform = dict()
        
        for DiterKey in DiterList:
            self.__vocabs[DiterKey] = dict()
            self.__text_transform[DiterKey] = dict()
            self.__D[DiterKey] = DiterFunc(DiterKey=DiterKey)
            for lang in self.langs:
                self.__vocabs[DiterKey][lang] = build_vocab_from_iterator(self.yield_tokens(self.__D[DiterKey], lang), specials=self.special_symbols, **self.build_vocab_params)
                self.__vocabs[DiterKey][lang].set_default_index(self.UNK_IDX)
                self.__text_transform[DiterKey][lang] = self.sequential_transforms(
                    getattr(self, f'spacy_{lang}', None), #Tokenization
                    self.__vocabs[DiterKey][lang], #Numericalization
                    self.tensor_transform # Add BOS/EOS and create tensor
                )
            self.__dataloaders[DiterKey] = getattr(DataModuleFromConfig(
                **{
                    DiterKey: {
                        'params': {'dataset': self.__D[DiterKey]}
                    }
                },
                batch_size=2,
                use_dck_mapper=False,
                dataset_category=[DiterKey],
                custom_collate=def_instance_method(self, f'_{DiterKey}_collate_fn', self.collate_fn, DiterKey=DiterKey)
            ), f'_{DiterKey}_dataloader')()
    
    def build_vocab(self, **build_vocab_params):
        """this function can be overwrite in child class"""
        build_vocab_params['min_freq'] = int(build_vocab_params.get('min_freq', 1))
        build_vocab_params['special_first'] = bool(build_vocab_params.get('special_first', True))
        # build_vocab_params['max_size'] = int(build_vocab_params.get('max_size', 10000))
        self.build_vocab_params = build_vocab_params
        DiterList = ['train', 'valid', 'test']
        def DiterFunc(DiterKey):
            return Multi30k(split=DiterKey, language_pair=self.langs)
        self.__build_vocab(DiterList, DiterFunc)