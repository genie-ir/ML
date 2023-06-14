from loguru import logger
import pytorch_lightning as pl
from libs.dyimport import Import
from torch.utils.data import random_split, DataLoader, Dataset

# def instantiate_from_config(config):
#     if not 'target' in config:
#         raise KeyError('Expected key `target` to instantiate.')
#     return Import(config['target'])(**config.get('params', dict()))

class WrappedDatasetBase(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& __getitem__')
        return self.data[idx]

class DataModuleFromConfigBase(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, wrap=False, num_workers=None, instantiate_from_config=None, custom_collate=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader # meaningfull name
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = self._val_dataloader # meaningfull name
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader # meaningfull name
        self.instantiate_from_config = instantiate_from_config
        self.custom_collate = custom_collate
        self.wrap = wrap
        self.wrap_cls = WrappedDatasetBase

    def _train_dataloader(self):
        # logger.warning('_train_dataloader is called!!!!!!!!!')
        if self.datasets.get('train', None) is None:
            return None
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.custom_collate)

    def _val_dataloader(self):
        # logger.warning('_val_dataloader is called!!!!!!!!!')
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def _test_dataloader(self):
        # logger.warning('_test_dataloader is called!!!!!!!!!')
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def prepare_data(self): # I think this function shoulde be remove since in setup function all datasets instantioation are done! TODO
        return
        # for data_cfg in self.dataset_configs.values():
        #     print('prepare_data', data_cfg)
        #     self.instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, self.instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs) # self.datasets contain datasets such as imageNetTrain, imageNetValidation, ... and so on.
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = self.wrap_cls(self.datasets[k])

