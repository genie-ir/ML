try:
    import os
    from apps.VQGAN.data.utils import custom_collate
    from apps.VQGAN.modules.configuration import Config
    from utils.pl.plDataset import DataModuleFromConfigBase, WrappedDatasetBase
except Exception as e:
    print(e)
    assert False

from libs.basicIO import extractor


class WrappedDataset(WrappedDatasetBase):
    pass

class DataModuleFromConfig(DataModuleFromConfigBase):
    def __init__(self, **kwargs):
        # kwargs['wrap']=True # this line it shoulde be remove. TODO
        # os.system('kaggle datasets download -d andrewmvd/drive-digital-retinal-images-for-vessel-extraction')
        extractor('/content/drive2004.zip', '/content/dataset_drive')
        kwargs['custom_collate'] = custom_collate
        kwargs['instantiate_from_config'] = Config.instantiate_from_config
        super().__init__(**kwargs)
        self.wrap_cls = WrappedDataset
