from loguru import logger
from os.path import join, exists
from libs.basicIO import download, file_hash as md5_file_hash

def get_ckpt_path(name, root, check=False, **kwargs):
    logger.warning('utils>tools.py>get_ckpt_path | [utils.tools.py] $$$$$$$$$$$$$$$$$$$$$', kwargs)
    URL_MAP = kwargs.get('URL_MAP', dict())
    MD5_MAP = kwargs.get('MD5_MAP', dict())
    CKPT_MAP = kwargs.get('CKPT_MAP', dict())
    
    assert name in URL_MAP, f'name: {name} is not in `URL_MAP`.'
    path = join(root, CKPT_MAP[name])
    if not exists(path) or (check and not md5_file_hash(path) == MD5_MAP[name]):
        print('Downloading {} model from {} to {}'.format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_file_hash(path)
        assert md5 == MD5_MAP[name], f'md5: `{md5}` is not correct hash.'
    return path