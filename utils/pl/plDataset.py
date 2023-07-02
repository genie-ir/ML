from loguru import logger
import pytorch_lightning as pl
from libs.dyimport import Import
from libs.basicDS import def_instance_method
from torch.utils.data import random_split, DataLoader, Dataset

# def instantiate_from_config(config):
#     if not 'target' in config:
#         raise KeyError('Expected key `target` to instantiate.')
#     return Import(config['target'])(**config.get('params', dict()))

class WrappedDatasetBase(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, D):
        self.D = D

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx]

class DataModuleFromConfigBase(pl.LightningDataModule):
    def __init__(self, batch_size, wrap=False, num_workers=None, instantiate_from_config=None, custom_collate=None, wrap_cls=None, **kwargs):
        super().__init__()

        kwargs['dataset_category'] = list(kwargs.get('dataset_category', ['train', 'test', 'validation']))

        self.kwargs = kwargs
        self.datasets = dict()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        
        self.wrap = bool(wrap)
        self.custom_collate = custom_collate
        self.wrap_cls = wrap_cls or WrappedDatasetBase
        self.instantiate_from_config = instantiate_from_config
        
        self.dataset_configs = dict((self.dck_mapper(dck), self.kwargs.get(dck, None)) for dck in self.kwargs['dataset_category'])
        
        print('----------------------->', self.dataset_configs, list(self.dataset_configs.keys()))

        for DCK, DCV in self.dataset_configs.items():
            if DCV is not None:
                SPECIFIC_PL_FN = f'{DCK}_dataloader' # automaticaly call by pytorch_lightning
                setattr(self, SPECIFIC_PL_FN, getattr(self, f'_{SPECIFIC_PL_FN}', def_instance_method(self, f'_{SPECIFIC_PL_FN}', self._dataloader, DCK=DCK))) # this is called by pytorch_lightning automatically.
                self.datasets[DCK] = self.instantiate_from_config(DCV)
                if self.wrap:
                    self.datasets[DCK] = self.wrap_cls(self.datasets[DCK])

        # if train is not None:
        #     self.dataset_configs['train'] = train
        #     self.train_dataloader = self._train_dataloader # meaningfull name
        # if validation is not None:
        #     self.dataset_configs['validation'] = validation
        #     self.val_dataloader = self._val_dataloader # meaningfull name
        # if test is not None:
        #     self.dataset_configs['test'] = test
        #     self.test_dataloader = self._test_dataloader # meaningfull name
        

    def dck_mapper(self, dck):
        dck = str(dck).lower()
        for _category in ['train', 'test', 'val']:
            category = str(_category).lower()
            if category in dck:
                return category
        return dck
    
    def _dataloader(self, **kwargs):
        print('##############', kwargs, self.datasets[kwargs['memory']['DCK']])
        return DataLoader(self.datasets[kwargs['memory']['DCK']], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.custom_collate)
    
    # def _train_dataloader(self):
    #     # logger.warning('_train_dataloader is called!!!!!!!!!')
    #     if self.datasets.get('train', None) is None:
    #         return None
    #     return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.custom_collate)

    # def _val_dataloader(self):
    #     # logger.warning('_val_dataloader is called!!!!!!!!!')
    #     return DataLoader(self.datasets['validation'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    # def _test_dataloader(self):
    #     # logger.warning('_test_dataloader is called!!!!!!!!!')
    #     return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    # def __setup(self, stage=None):
        
    #     self.datasets = dict(
    #         (dck, self.instantiate_from_config(self.dataset_configs[dck]))
    #         for dck in self.dataset_configs) # self.datasets contain datasets such as imageNetTrain, imageNetValidation, ... and so on.
        
        
    #     if self.wrap:
    #         for k in self.datasets:
    #             self.datasets[k] = self.wrap_cls(self.datasets[k])

