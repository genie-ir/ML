from .dataset import EYEQ, DEEPDR, DRAC, IQAD_CXR, IQAD_CT
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_dataloader(cfg, **kwargs):
    # print(cfg)
    root = {
        'ROOT': cfg.DATASET.ROOT,
        'DATADIR': cfg.DATASET.DATADIR,
        'MAPSPLIT': {
            'val': '/content/DeepDRiD/regular_fundus_images/regular-fundus-validation',
            'test': '/content/DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation',
            'train': '/content/DeepDRiD/regular_fundus_images/regular-fundus-training'
        }
    }
    batch_size = cfg.BATCH_SIZE
    dataset_name = cfg.DATASET.NAME

    # print('@@@@@@@@', root)

    train_ts, test_ts = get_transform(cfg)
    num_worker = 2
    dataset = globals()[dataset_name]
    # print('------------------->', dataset)

    train_dataset = dataset(root=root, split = 'train', transform=train_ts, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers= num_worker)

    if cfg.DATASET.NAME in cfg.VALIDATION_DATASET:
        val_dataset = dataset(root=root, split = 'val', transform=test_ts, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    test_dataset = dataset(root=root, split = 'test', transform=test_ts, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    dataset_size = [len(train_dataset), len(test_dataset)]
    
    if dataset_name in cfg.VALIDATION_DATASET:
        return train_loader, test_loader, val_loader, dataset_size
    else:
        return train_loader, test_loader, dataset_size

def get_transform(cfg):

    means = cfg.DATASET.NORMALIZATION_MEAN
    std = cfg.DATASET.NORMALIZATION_STD

    transfrom_train = []
    transfrom_test = []

    # if dataset in ['DRAC', 'IQAD_CXR', 'IQAD_CT']:
    #     transfrom_train.append(transforms.Grayscale(1))
    #     transfrom_test.append(transforms.Grayscale(1))

    transfrom_train.append(A.LongestMaxSize(256))
    transfrom_test.append(A.LongestMaxSize(256))

    transfrom_train.append(A.Normalize(means, std))
    transfrom_train.append(ToTensorV2())

    transfrom_test.append(A.Normalize(means, std))
    transfrom_test.append(ToTensorV2())

    train_ts =  A.Compose(transfrom_train)
    test_ts = A.Compose(transfrom_test)

    return train_ts, test_ts
