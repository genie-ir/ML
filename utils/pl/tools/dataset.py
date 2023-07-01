from libs.dyimport import instantiate_from_config
print('000000000000000000000000000000000000')
from utils.pl.tools.dataset_utils import custom_collate
print('111111111111111111111111111111111111')
from utils.pl.plDataset import DataModuleFromConfigBase, WrappedDatasetBase
print('22222222222222222222222222222222222')

class WrappedDataset(WrappedDatasetBase):
    pass

class DataModuleFromConfig(DataModuleFromConfigBase):
    def __init__(self, **kwargs):
        # kwargs['wrap']=True # this line it shoulde be remove. TODO

        kwargs['custom_collate'] = custom_collate
        kwargs['instantiate_from_config'] = instantiate_from_config
        super().__init__(**kwargs)
        self.wrap_cls = WrappedDataset
