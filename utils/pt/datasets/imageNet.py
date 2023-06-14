import os, tarfile, glob, shutil, sys, cowsay
from os import makedirs, system, environ, getenv, link, rename
from os.path import join, exists, relpath, getsize, abspath
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from libs.coding import sha1
from libs.basicIO import dffilter, dfread, dfwrite, fwrite, extractor, pathBIO, download, is_prepared, mark_prepared, dotdict
from libs.basicDS import retrieve

# from apps.VQGAN.util import retrieve

class ImageNetBase(Dataset):
    """
        relpaths:     [... '/x1.png' ...]  It can be consider as `x`
        synsets:      [... 'class_0' ...]
        class_labels: [...     0     ...]  It can be consider as `y`

    """
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        
        self.BYPASS_SYNSETS_FLAG = bool(self.config.get('BYPASS_SYNSETS_FLAG', False))

        ################[TODO]################
        # int   -> use for classification [ok]
        # float -> use for regression     [need to do code]
        # str   -> use for classification [need to do code]
        self.DF_CLASS_TYPE = self.config.get('DF_CLASS_TYPE', 'int')
        assert self.DF_CLASS_TYPE in ['int'], '[TODO]: we must do code for DF_CLASS_TYPE={}'.format(self.DF_CLASS_TYPE)
        ######################################

        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def download_dataset(self, **kwargs):
        if kwargs.get('api', None) is None:
            cowsay.cow('NotImplementedError:\nplease define `{}:download_dataset` function or just define `api` command at config file.'.format(self.__class__.__name__))
            sys.exit()
        else:
            apicmd = kwargs['api']
            for k, v in kwargs.items():
                apicmd = apicmd.replace('$' + k, str(v))
            print(self.__class__.__name__, apicmd)
            os.system(apicmd)

    def _prepare(self):
        self.HOST_DIR = self.config['HOST_DIR']
        if self.HOST_DIR.upper() == '$KAGGLE_PATH':
            self.HOST_DIR = pathBIO('//' + getenv('KAGGLE_PATH'))
        if self.HOST_DIR.startswith('//'):
            self.HOST_DIR = pathBIO(self.HOST_DIR)

        makedirs(self.HOST_DIR, exist_ok=True)

        self.NAME = self.config['NAME']
        self.FILES = self.config.get('FILES', [])
        # self.root = join(cachedir, 'autoencoders/data', self.NAME)
        self.root = join(pathBIO(getenv('GENIE_ML_CACHEDIR')), self.NAME)
        self.datadir = join(self.root, 'data')
        self.hashdir = join(self.root, 'hash')
        makedirs(self.hashdir, exist_ok=True)
        self.txt_filelist_head = self.config.get('txt_filelist', 'filelist.txt')
        self.txt_filelist = join(self.root, self.txt_filelist_head)
        self.readyFnameAttr = '.ready-' + self.txt_filelist_head.replace('.', '__')

        TEMP_NAME_DATASET = self.__class__.__name__
        TEMP_NAME_DATAFRM = self.config.get('DF_NAME', None)
        assert not (TEMP_NAME_DATASET is None or TEMP_NAME_DATAFRM is None), 'TEMP_NAME_DATASET={} | TEMP_NAME_DATAFRM={} | both of these variables it must be not None'.format(TEMP_NAME_DATASET, TEMP_NAME_DATAFRM)
        self.filtered_filelist = join(self.root, '{}__filtered_filelist.npy'.format(TEMP_NAME_DATASET))
        self.synsets_of_filtered_filelist = join(self.root, '{}__synsets_of_filtered_filelist.npy'.format(TEMP_NAME_DATASET))
        self.df_path = join(self.datadir, TEMP_NAME_DATAFRM)
        self.df_candidate_path = join(self.datadir, 'candidate_' + TEMP_NAME_DATAFRM)
        
        self.preparation() # this function is overwrite by the user.

        if not is_prepared(self.root, fname=self.readyFnameAttr):
            logger.info('{} | Preparing dataset {} in {}'.format(self.__class__.__name__, self.NAME, self.root))
            datadir = self.datadir
            makedirs(datadir, exist_ok=True)
            for fname in self.FILES:
                fake_fpath = join(self.root, fname)
                if not exists(fake_fpath):
                    real_fdir = join(self.HOST_DIR, self.NAME)
                    real_fpath = join(real_fdir, fname)
                    real_fpath = (glob.glob(real_fpath + '*') + [real_fpath])[0]
                    if not exists(real_fpath):
                        makedirs(real_fdir, exist_ok=True)
                        makedirs(getenv('GENIE_ML_STORAGE0'), exist_ok=True)
                        self.download_dataset(real_fdir=real_fdir, api=self.config.get('api', None))
                        real_fpath = glob.glob(real_fpath + '*')[0]
                    
                    print('real_fpath', real_fpath)
                    print('fake_fpath', fake_fpath)
                    link(src=real_fpath, dst=fake_fpath)
                
                hashbased_path = join(self.hashdir, sha1(fake_fpath))
                if not exists(hashbased_path):
                    try:
                        makedirs(hashbased_path, exist_ok=True)
                        self.extract_dataset(fake_fpath=fake_fpath, datadir=datadir)
                    except Exception as e:
                        logger.error(e)

            if self.config.get('N_WSTAR', 1) == -1:
                inpalceWSTAR = self.config.get('S_WSTAR', '')
            else:
                inpalceWSTAR = join(*['**' for istar in range(self.config.get('N_WSTAR', 1))])
            filelist = glob.glob(join(datadir, inpalceWSTAR, '*.{}'.format(self.config['EXT']))) # inside datadir we scaped one level directories and we select specefic `.EXT` files
            filelist = [relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = '\n'.join(filelist) + '\n'
            with open(self.txt_filelist, 'w') as f:
                f.write(filelist)

            mark_prepared(self.root, fname=self.readyFnameAttr)
        
        self.DF_KEY = self.config.get('DF_KEY', None) 
        self.DF_VAL = self.config.get('DF_VAL', None)
        assert not (self.DF_KEY is None or self.DF_VAL is None), 'DF_KEY={} | DF_VAL={} | both variables it must be not None'.format(self.DF_KEY, self.DF_VAL)
        if exists(self.df_path):
            self.df = dfread(self.df_path)
        else:
            self.df = pd.DataFrame({self.DF_KEY: [], self.DF_VAL: []})
        self.DF_CANDIDATE = self.config.get('DF_CANDIDATE', dict())
        # `example_id` is what that we can select one row from dataframe.
        # if the row doesn't exist we return `None` otherwise we return `class value` from specefic column.
        self.classLableValueFn = lambda example_id: (list(self.df.loc[self.df[self.DF_KEY]==example_id][self.DF_VAL]) + [None])[0]

    def preparation(self, **kwargs):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths, cb=None, ignore_list=None):
        if exists(self.filtered_filelist):
            return np.load(self.filtered_filelist)

        # Example: ignore_list = ['n06596364_9591.JPEG', ...]
        ignore_list = ignore_list if ignore_list else []
        ignore = set(ignore_list)
        cb = cb if cb else lambda inp: True
        _cb = lambda _inp: bool(cb(_inp) and (not _inp.split('/')[-1] in ignore))
        relpaths = [rpath for rpath in tqdm(relpaths, desc='filtering of relpaths list') if _cb(rpath)]
        
        mode_val = self.config.get('MODE_VAL', None)
        if isinstance(mode_val, int):
            relpaths = [rpath for idx, rpath in tqdm(enumerate(relpaths), desc='filtering of relpaths list through indexing with mode={}'.format(mode_val)) if idx % mode_val == 0]

        np.save(self.filtered_filelist, relpaths)
        return relpaths
        if 'sub_indices' in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        self.human_dict = join(self.root, '{}__synset_human.txt'.format(self.__class__.__name__))
        if not exists(self.human_dict):
            download(self.config['URL']['synset'], self.human_dict, copyFlag=True)

    def _prepare_idx_to_synset(self):
        self.idx2syn = join(self.root, '{}__index_synset.yaml'.format(self.__class__.__name__))
        if not exists(self.idx2syn):
            if str(self.config['URL']['iSynset']).startswith('$'):
                iSynset_file_content = '\n'.join([f'{i}: class_{i}' for i in range(int(self.config['URL']['iSynset'][1:]))])
                fwrite(self.idx2syn, iSynset_file_content)
            else:
                download(self.config['URL']['iSynset'], self.idx2syn, copyFlag=True) # TODO

    def accept_reject_callback(self, inp):
        """
            This function can be `overwrite` in child class for specefic perpose
            and It must be return True/False.
            if return `True` then we `accept` coresponding row, otherwise we reject that row. 
        """
        # if classification task then use int TODO
        return isinstance(self.classLableValueFn(inp.split(os.sep)[-1]), (int, float))
    
    def set_tag_to_record(self, record_path, record_fname):
        """
            This function can be `overwrite` in child class for specefic perpose
            and It be useful in some cases like that there is no datafram for label assignment.
        """
        PATH_TO_CLASS_MAP = self.config.get('PATH_TO_CLASS_MAP', dict())
        assert record_path in PATH_TO_CLASS_MAP, '->{}<-, dict={}'.format(record_path, PATH_TO_CLASS_MAP)
        return PATH_TO_CLASS_MAP.get(record_path, -1)

    def _load(self):
        with open(self.txt_filelist, 'r') as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            if len(self.df) == 0:
                for _rp in tqdm(self.relpaths, desc='creating DataFrame -> {}'.format(self.df_path)):
                    rp = abspath(join(os.sep, _rp))
                    rps = rp.split(os.sep)
                    self.df = pd.concat([self.df, pd.DataFrame.from_records([
                        {self.DF_KEY: rps[-1], self.DF_VAL: self.set_tag_to_record(os.sep.join(rps[:-1]), rps[-1])}
                    ])])
                self.df = self.df.astype({self.DF_VAL: eval(self.DF_CLASS_TYPE)})
                self.df = self.df.reset_index(drop=True)
                dfwrite(self.df_path, self.df)
            
            if self.DF_CANDIDATE is None:
                self.DF_CANDIDATE = dict()
            if len(list(self.DF_CANDIDATE.keys())) > 0:
                if exists(self.df_candidate_path):
                    self.df = dfread(self.df_candidate_path)
                else: # Here we reject some records from self.df and only retain candidate rows.
                    self.df = dffilter(self.df, 
                        MAX_N_IN_CAT=[self.DF_CANDIDATE.get('MAX_N_IN_CAT', float('inf')), self.DF_VAL]
                    )
                    dfwrite(self.df_candidate_path, self.df)
            self.relpaths = self._filter_relpaths(self.relpaths, cb=self.accept_reject_callback)
            logger.info('{} | ({}/{}) -> Removed {} files from filelist during filtering.'.format(self.__class__.__name__, len(self.relpaths), l1, l1 - len(self.relpaths)))

        if exists(self.synsets_of_filtered_filelist):
            self.synsets = np.load(self.synsets_of_filtered_filelist)
        else:
            self.synsets = ['class_' + str(self.classLableValueFn(p.split('/')[-1])) for p in tqdm(self.relpaths, desc='creation of synsets list')]
            np.save(self.synsets_of_filtered_filelist, self.synsets)
        logger.info('{} | len(relpaths)={}, len(Synset)={}'.format(self.__class__.__name__, len(self.relpaths), len(self.synsets)))
        # self.abspaths = [join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        logger.info('{} | unique_synsets: {}'.format(self.__class__.__name__, unique_synsets))
        # class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        class_dict = dict()
        class_dict_R = dict()
        with open(self.idx2syn, 'r') as f:
            idx_dict = f.read().splitlines()
            for line in idx_dict:
                line_strip = line.strip()
                if line_strip:
                    line_strip_split = line_strip.split(':')
                    if len(line_strip_split) == 2:
                        temp_key_i = line_strip_split[0].strip()
                        temp_val_i = line_strip_split[1].strip()
                        class_dict[temp_val_i] = temp_key_i
                        if temp_key_i in class_dict_R:
                            if isinstance(class_dict_R[temp_key_i], list):
                                class_dict_R[temp_key_i].append(temp_val_i)
                            else:
                                class_dict_R[temp_key_i] = [class_dict_R[temp_key_i]]
                                class_dict_R[temp_key_i].append(temp_val_i)
                        else:
                            class_dict_R[temp_key_i] = temp_val_i
        try:
            self.class_labels = [eval('{}({})'.format(
                self.DF_CLASS_TYPE, class_dict[s]
            )) for s in self.synsets]
        except KeyError as ke:
            assert False, 'class `{}` Does not support by `{}`'.format(ke, self.idx2syn)
        except Exception as e:
            assert False, 'ERROR: {}'.format(e)
        
        logger.info('{} | class_dict={}'.format(self.__class__.__name__, class_dict))
        
        with open(self.human_dict, 'r') as f:
            human_dict = f.read().splitlines()
            assert not ('|' in '\n'.join(human_dict)), 'char `|` should not be in file: `{}`'.format(self.human_dict)
            human_dict = dict(line.strip().split(maxsplit=1) for line in human_dict if line.strip())
            human_dict_R = dict()
            for CDR_k, CDR_v in class_dict_R.items():
                if isinstance(CDR_v, list):
                    human_dict_R[CDR_k] = ' | '.join([human_dict[cdrv_i] for cdrv_i in CDR_v])
                else:
                    human_dict_R[CDR_k] = human_dict[CDR_v]

        logger.info('{} | human_dict: {}'.format(self.__class__.__name__, human_dict))
        # self.human_labels = [human_dict[s] for s in self.synsets]

        assert len(self.relpaths) == len(self.synsets) == len(self.class_labels), 'These lengths should be equal | len(relpaths)={} | len(synsets)={} | len(class_labels)={}'.format(len(self.relpaths), len(self.synsets), len(self.class_labels))
        
        labels = { # (#:ignore) (@:function(y)) ($:function(x))
            'x': np.array(self.relpaths),
            'y': np.array(self.class_labels),
            '#upper_path': self.datadir
        }

        # print(self.relpaths)
        # print(self.class_labels)
        # print(class_dict)
        # print(class_dict_R)
        # print(human_dict_R)
        
        if self.DF_CLASS_TYPE in ('int', 'str'): # classification task
            labels['@human_label'] = lambda yi: human_dict_R[str(yi)]
        if self.BYPASS_SYNSETS_FLAG:
            labels['synsets'] = np.array(self.synsets)
        
        self.data = self.D(
            labels=labels,
            **self.config,
            size=retrieve(self.config, 'SIZE', default=0)
        )

class ImageNetTrain(ImageNetBase):
    """
    it useful to overide functions below in creation of custom dataset:
        `download_dataset`, `extract_dataset`, ``preparation` of parrent class`
    """
    def extract_dataset(self, **kwargs):
        fake_fpath = kwargs['fake_fpath']
        datadir = kwargs['datadir']
        extractor(src_file=fake_fpath, dst_dir=datadir, mode='zip')
        nested_list = glob.glob(join(datadir, '*.zip*'))
        assert len(nested_list)==0, f'nested_list: {nested_list} is exist.'

class ImageNetValidation(ImageNetBase):
    """
    it useful to overide functions below in creation of custom dataset:
        `download_dataset`, `extract_dataset`, `preparation of parrent class`
    """
    def extract_dataset(self, **kwargs):
        fake_fpath = kwargs['fake_fpath']
        datadir = kwargs['datadir']
        extractor(src_file=fake_fpath, dst_dir=datadir, mode='zip')
        nested_list = glob.glob(join(datadir, '*.zip*'))
        assert len(nested_list)==0, f'nested_list: {nested_list} is exist.'