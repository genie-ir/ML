import os
import sys
import time
import yaml
import json
import glob
import torch
import boto3
import random
import pickle
import shutil
import ftplib
import pathlib
import zipfile
import tarfile
import zipfile
import libtmux
import requests
import subprocess
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from time import sleep
from torf import Torrent
from loguru import logger
from libs.basicHR import EHR
from omegaconf import OmegaConf
from os.path import join, exists
from os import getenv, symlink, link
from libs.basicTime import getTimeHR_V0
from libs.basicDS import dict2ns, dotdict
from libs.coding import md5, sha1, random_string

def rootPath():
    return pathlib.Path(__file__).parents[1]

def pathBIO(fpath: str, **kwargs):
    if fpath.startswith('//'):
        fpath = join(rootPath(), fpath[2:])
    return fpath

def puml(src_fname: str, dst_fname: str, **kwargs):
    assert src_fname.endswith('.puml'), '`src_fname={}` | It must be ends with `.puml`'.format(src_fname)
    d = kwargs.get('d', None)
    u = kwargs.get('u', None)
    src = join(kwargs.get('src_dpath', getenv('GENIE_ML_REPORT')), src_fname)
    
    if not (d is None):
        if isinstance(d, dict):
            dd = d
        elif isinstance(d, type(OmegaConf.create())):
            dd = OmegaConf.to_container(d)
        else:
            assert False, '`type(dd)={}` | currently doesnt support. plese `implement` code for this'.format(type(dd))
        fwrite(src, dict2strjson(dd, s=kwargs.get('s', '@startjson\n'), e=kwargs.get('e', '\n@endjson')))

    if not (u is None):
        assert isinstance(u, str), '`type(u)={}` | It must be `str`'.format(type(u))
        fwrite(src, kwargs.get('s', '@startuml\n') + u + kwargs.get('e', '\n@enduml'))
    
    os.system('cat "{}" | java -jar "{}" -pipe > "{}"'.format(
        src,
        kwargs.get('jarpath', 
            # join(getenv('GENIE_ML_STORAGE0'), '..', '..', 'plantuml.jar')
            pathBIO('//' + join('dependency', 'plantuml.jar'))
        ),
        join(kwargs.get('dst_dpath', getenv('GENIE_ML_REPORT')), dst_fname)
    ))

def pklread(src: str):
    if not src.endswith('.pkl'):
        src = src + '.pkl'
    return pickle.load(open(src, 'rb'))

def pklwrite(dst: str, data):
    if not dst.endswith('.pkl'):
        dst = dst + '.pkl'
    pickle.dump(data, open(dst, 'wb'))

def fwrite(dst: str, content, mode='w'):
    f = open(dst, mode)
    f.write(content)
    f.close()

def dfwrite(dst: str, df, **kwargs):
    if dst.endswith('.csv'):
        df.to_csv(dst, sep=kwargs.get('sep', ','), encoding=kwargs.get('encoding', 'utf-8'), index=kwargs.get('index', False))

def dfread(src: str, **kwargs):
    pattern = str(kwargs.get('pattern', ''))
    if pattern:
        print(ls(src, pattern, full_path=True))
    else:
        if src.endswith('.csv'):
            return pd.read_csv(src)

def dfshuffle(df, frac=1.0, resetIndexFlag=True):
    random_state = int(int(round(time.time() * 1000)) % 10000)
    if resetIndexFlag:
        return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    else:
        return df.sample(frac=frac, random_state=random_state)

def dffilter(df, **kwargs):
    kwargs_keys = list(kwargs.keys())
    for k in kwargs_keys:
        # if k == '...':
        #     pass
        if k == 'MAX_N_IN_CAT' and isinstance(kwargs[k], (list, tuple)) and len(kwargs[k]) == 2:
            if kwargs[k][0] < float('inf'):
                df = dfshuffle(pd.concat([
                    dfshuffle(df[df[str(kwargs[k][1])]==uv])[:int(kwargs[k][0])]
                    for uv in tqdm(df[str(kwargs[k][1])].unique(), desc='dffilter [MAX_{}_IN_CAT]'.format(int(kwargs[k][0])))
                ]))
    return df

def dict2strjson(d, **kwargs):
    json.dumps(d)
    s = str(kwargs.get('s', ''))
    e = str(kwargs.get('e', ''))
    return f'{s}{json.dumps(d)}{e}' 

class YAML_Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(YAML_Loader, self).__init__(stream)

    @staticmethod
    def addc():
        if getattr(YAML_Loader, 'flag_addc', True):
            YAML_Loader.add_constructor('!join', YAML_Loader.join)
            YAML_Loader.add_constructor('!merge', YAML_Loader.merge)
            YAML_Loader.add_constructor('!include', YAML_Loader.include)
            YAML_Loader.flag_addc = False
    
    @staticmethod
    def addifc(ifc):
        if ifc:
            YAML_Loader.instantiate_from_config = ifc

    def join(self, node):
        seq = self.construct_sequence(node)
        return '.'.join([str(i) for i in seq])
    
    def merge(self, node):
        seq = self.construct_sequence(node)
        out = dict()
        for s in seq:
            out = {**out, **s}
        return out

    def include(self, node):
        if isinstance(node, yaml.nodes.ScalarNode):
            seq = [self.construct_scalar(node)]
        elif isinstance(node, yaml.nodes.SequenceNode):
            seq = self.construct_sequence(node)
        else:
            assert False, '`type(node)={}` | does not supported'.format(type(node))

        filename = os.path.join(seq[0])
        if not filename.endswith('.yaml'):
            filename = filename + '.yaml'
        out =  YAML_Loader.instantiate_from_config({'target': filename}, kwargs={'dotdictFlag': False, 'Loader': YAML_Loader})
        if len(seq) > 1:
            return {str(seq[1]): out}
        else:
            return out

def readBIO(fpath: str, **kwargs):
    """
        yamlFile = readBIO('/test.yaml')
        jsonFile = readBIO('/test.json')
    """
    fpath = pathBIO(fpath)
    ext = fpath.split('.')[-1].lower()
    dotdictFlag = kwargs.get('dotdictFlag', None)

    if ext == 'yaml':
        try:
            YAML_Loader.addc()
            YAML_Loader.addifc(kwargs.get('instantiate_from_config', None))
            with open(fpath, 'r') as f:
                out = yaml.load(f, Loader=kwargs.get('Loader', YAML_Loader))
                if (isinstance(out, dict)) and ('$vars' in out):
                    del out['$vars']
                if (isinstance(out, dict)) and ('$merge' in out):
                    merge_dict = {**out['$merge']}
                    del out['$merge']
                    out = OmegaConf.to_container(OmegaConf.merge(*[merge_dict, out]))
                return dotdict(out, flag=dotdictFlag)
        except Exception as e:
            EHR(e)
    
    if ext == 'json':
        try:
            with open(fpath) as f:
                return dotdict(json.load(f), flag=dotdictFlag)
        except Exception as e:
            EHR(e)

def check_logdir(fpath: str, **kwargs):
    fpath = pathBIO(fpath)
    if not exists(fpath): # TODO AND fpath INSIDE LOG DIR.
        pass 

def ls(_dir, _pattern: str, full_path=False):
    """
        print(glob.glob('/home/adam/*.txt'))   # All files and directories ending with .txt and that don't begin with a dot:
        print(glob.glob('/home/adam/*/*.txt')) # All files and directories ending with .txt with depth of 2 folders, ignoring names beginning with a dot:
    Example:
        print(ls('/', '*.jpg'))
    """
    if isinstance(_dir, dict):
        return list(_dir.keys()) # _dir contains multi directory informations.
    assert isinstance(_dir, str), f'_dir is must be str (path directory). but now is {type(_dir)}'
    
    _dir = pathBIO(_dir)

    if full_path:
        return glob.glob(join(_dir, _pattern))
    else:
        return glob.glob1(_dir, _pattern) # Notic: glob1 does not support (**) (match child dirs)

def is_prepared(adr, fname='.ready'):
    return pathlib.Path(adr).joinpath(fname).exists()

def mark_prepared(adr, fname='.ready'):
    pathlib.Path(adr).joinpath(fname).touch()

def compressor(src_dir: str, dst_file: str, mode=None):
    if mode is None:
        if dst_file.endswith('.tar'):
            mode = 'tar'
        elif dst_file.endswith('.zip'):
            mode = 'zip'

    if mode == 'tar':
        pass
    
    if mode == 'zip':
        with zipfile.ZipFile(dst_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for sd_root, sd_dirs, sd_files in os.walk(src_dir):
                for sd_file in sd_files:
                    zipf.write(join(sd_root, sd_file), os.path.relpath(join(sd_root, sd_file), join(src_dir, '..')))


def extractor(src_file, dst_dir, mode='tar', delFlag=False, makeReadyFlag=False):
    assert not is_prepared(dst_dir), 'dst_dir=`{}` is already exist'.format(dst_dir)
    flag = False
    if mode == 'tar':
        flag = True
        with tarfile.open(src_file, 'r:') as tar_ref:
            # tar_ref.extractall(path=dst_dir)
            for file in tqdm(iterable=tar_ref.namelist(), total=len(tar_ref.namelist()), desc='extracting {}'.format(src_file)):
                # Extract each file to another directory
                # If you want to extract to current working directory, don't specify path
                try:
                    tar_ref.extract(member=file, path=dst_dir)
                except Exception as e:
                    logger.error(e)
    
    if mode == 'zip':
        flag = True
        with zipfile.ZipFile(src_file, 'r') as zip_ref:
            # zip_ref.extractall(dst_dir)
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc='extracting {}'.format(src_file)):
                # Extract each file to another directory
                # If you want to extract to current working directory, don't specify path
                try:
                    zip_ref.extract(member=file, path=dst_dir)
                except Exception as e:
                    logger.error(e)
    
    assert flag, 'mode=`{}` is not supported'.format(mode)
    if delFlag:
        # os.system('sudo rm -rf {}'.format(src_file))
        print('[deleting] -> {}'.format(src_file))
    if makeReadyFlag:
        mark_prepared(dst_dir)

def download(url: str, local_path, chunk_size=1024, copyFlag=False):
    """HTTP download function"""
    if exists(local_path):
        if getenv('GENIE_ML_DEBUG_MODE') == 'True':
            logger.debug('local_path already is exist.')
        return

    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    if not url.startswith('http'): # Notic: https also strats with http
        url = pathBIO(url)
        if copyFlag:
            shutil.copyfile(url, local_path)
        else:
            link(src=url, dst=local_path)
        return

    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def file_hash(path, fn=md5):
    with open(path, 'rb') as f:
        content = f.read()
    return fn(content)

def __signal_save__img_Tensor(images, fpath, nrow=None, fn=None):
        if fn is None:
            fn = lambda G: G
        nrow = images.shape[0] if nrow is None else nrow
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
        grid = torchvision.utils.make_grid(images, nrow=nrow) # this grid finally contains table of iamges like this -> [images[k].shape[0]/nrow, nrow] ; Notic: grid is tensor with shape: ch x h? x w?
        grid = fn(grid).numpy().astype(np.uint8)
        signal_save(grid, fpath)

def signal_save(s, path, makedirsFlag=True, stype=None, sparams=None):
    sparams = dict() if sparams is None else sparams
    assert isinstance(sparams, dict), '`type(sparams)={}` | It must be dict | looking this maybe useful: `sparams={}`'.format(type(sparams), sparams)
    path = pathBIO(path)
    dpath, fname = os.path.split(path)
    fname_lower = fname.lower()
    if makedirsFlag:
        os.makedirs(dpath, exist_ok=True)
    
    if isinstance(s, torch.Tensor):
        if stype == 'img':
            return __signal_save__img_Tensor(s, path, **sparams)
        s = s.cpu().detach().numpy()
    
    if isinstance(s, np.ndarray): # image signal
        if any(ext.lower() in fname_lower for ext in ['.png', '.jpg', '.jpeg']):
            return Image.fromarray(s).save(path)
        if any(ext.lower() in fname_lower for ext in ['.npy']):
            return np.save(path, s)
    
def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size # bytes

def copy_dir(_src, _dst, waitFlag=False, desc=None):
    src, dst = join(_src), join(_dst)
    
    if waitFlag:
        src_size = get_size(src)
        desc = desc if desc else 'copying from {} to {}'.format(src, dst)
        p_bar = tqdm(range(src_size), desc=desc)
        shutil.copytree(src, dst)
        dst_size = 0
        while(dst_size != src_size):
            dst_size = get_size(dst)
            p_bar.n = dst_size
            p_bar.refresh()
            sleep(1)
    else:
        shutil.copytree(src, dst)
            
def merge_files(src, dst, waitFlag=False, extraction_mode=None, desc=None, delFlag=True):
    """
        [Notic]: src is a splited directory path (input)
        [Notic]: dst is merged file path (goal)
        [Notic]: task of this function is merging `src` directory to `dst` file. | `dst` file usually is zipped file.
        [Notic]: this function can `optionally` extract `dst` file to `dst_extracted_dir`
    """
    assert exists(src), 'src=`{}` is not exist'.format(src)
    dst_extracted_dir = join(os.path.split(dst)[0], str(os.path.split(dst)[1]).replace('.', '__') + '__dir')
    flag = False
    
    if extraction_mode:
        assert waitFlag, 'waitFlag=`{}` is must be `True` when `extraction_mode` is used, now `extraction_mode` is `{}`'.format(waitFlag, extraction_mode)
        if is_prepared(dst_extracted_dir):
            return dst_extracted_dir
        if exists(dst):
            os.makedirs(dst_extracted_dir, exist_ok=True)
            extractor(dst, dst_extracted_dir, mode=extraction_mode, delFlag=delFlag, makeReadyFlag=True)
            return dst_extracted_dir
        else:
            flag = True # make `dst` and then `do extract it` and return extracted_dir path.
    else:
        if exists(dst):
            return dst
        else:
            flag = True # make `dst` and return its path.
    
    assert flag
    os.makedirs(os.path.split(dst)[0], exist_ok=True)
    if waitFlag:
        desc = desc if desc else 'merging from {} to {}'.format(src, dst)
        src_size = get_size(src)
        p_bar = tqdm(range(src_size), desc=desc)
        os.system('cat {}/* > {} &'.format(src, dst))
        dst_size = 0
        while(dst_size != src_size):
            try:
                dst_size = os.path.getsize(dst)
            except Exception as e:
                dst_size = 0
            p_bar.n = dst_size
            p_bar.refresh()
            sleep(1)
        if extraction_mode:
            os.makedirs(dst_extracted_dir, exist_ok=True)
            extractor(dst, dst_extracted_dir, mode=extraction_mode, delFlag=delFlag, makeReadyFlag=True)
            return dst_extracted_dir
        else:
            return dst
    else:
        os.system('cat {}/* > {} &'.format(src, dst))


class TorrentBase:
    def send(self, src, dst='my.torrent', private=True):
        
        self.t = Torrent(
            path=src
            # trackers=[
            #     'https://tracker1.example.org:1234/announce',
            #     'https://tracker2.example.org:5678/announce'
            # ],
            # comment='This is a comment'
        )
        self.t.private = private
        self.t.generate()
        self.t.write(dst)
        return self.t

class FTPBase:
    """
        Example:
            fb = FTPBase(...)
            print(fb.cd('/media/alihejrati/3E3009073008C83B/Rest/HOTD/S1'))
            print(fb.ls())
    """
    def __init__(self, user, passwd, host=None, port=21):
        host = host if host else os.uname()[1]
        self.server = ftplib.FTP()
        self.server.connect(str(host), int(port))
        self.server.login(str(user), str(passwd))
    
    def quit(self):
        """Send a QUIT command to the server and close the connection. This is the “polite” way to close a connection, but it may raise an exception if the server responds with an error to the QUIT command. This implies a call to the close() method which renders the FTP instance useless for subsequent calls (see below)."""
        return self.server.quit()
    
    def upload(self, src, dstdir='', tsFlag=True, tsFunction=getTimeHR_V0):
        dst = join(dstdir, os.path.split(src)[1])
        dst0, dst1 = os.path.split(dst)
        os.makedirs(dst0, exist_ok=True)
        if tsFlag:
            dst = join(dst0, tsFunction() + '_' + dst1)
        file = open(src, 'rb') # file to send
        res = self.server.storbinary('STOR {}'.format(dst), file) # send the file
        file.close() # close file and FTP
        return dst, res
    
    def ls(self):
        """ls directory"""
        return self.server.dir()

    def rename(self, fromname, toname):
        """Rename file fromname on the server to toname."""
        return self.server.rename(fromname, toname)
    
    def rm(self, dirname):
        """Remove the directory named `dirname` on the server."""
        return self.server.rmd(dirname)
    
    def rmf(self, filename):
        """Remove the file named `filename` from the server. If successful, returns the text of the response, otherwise raises error_perm on permission errors or error_reply on other errors."""
        return self.server.delete(filename)
    
    def mk(self, pathname):
        """Create a new directory on the server."""
        return self.server.mkd(pathname)

    def cd(self, pathname):
        """Set the current directory on the server."""
        return self.server.cwd(pathname)
    
    def pwd(self):
        """Return the pathname of the current directory on the server."""
        return self.server.pwd()
    
    def size(self, filename):
        """Request the size of the file named filename on the server. On success, the size of the file is returned as an integer, otherwise None is returned. Note that the SIZE command is not standardized, but is supported by many common server implementations"""
        return self.server.size(filename)
    
    def system(self, *psargs):
        cmd = str(' '.join(psargs)).strip()
        print('-> {}'.format(cmd))
        return self.server.sendcmd(cmd)
    
    def cb(self, f, *psargs, **kwargs):
        return getattr(self.server, f, lambda *p, **k: 'function name `{}` does not exist.'.format(f))(*psargs, **kwargs)

class SFTPBase:
    """
        Example:
            sb = SFTPBase(...)
            print(sb.cd('/media/alihejrati/3E3009073008C83B/Rest/HOTD/S1'))
            print(sb.ls())
    """
    def __init__(self, user, passwd, host=None, port=21):
        pass

class TMUXBase:
    """
        Example:
            tmux_obj = TMUXBase()
            pane = tmux_obj.get_pane(session_obj='teh', window_name='comp', pane_id='%9')
            tmux_obj.input(pane, "cowsay 'hello'") # ruu on the terminal
            tmux_obj.input(pane, "cowsay 'hello'", enter=False) # just type to terminal and not run it.
            out = tmux_obj.output(pane)
    """
    def __init__(self):
        os.system('tmux start-server')
        self.server = libtmux.Server()
        self.__sessions = dict() # self_created_sessions not all sessions. # just stores created sessions by this instance of class.

    def get_server(self):
        return self.server
    
    def get_all_sessions(self):
        return self.server.sessions
    
    def get_all_sessions_names(self):
        all_sessions = self.get_all_sessions()
        return [si.name for si in all_sessions]

    def get_self_created_sessions(self, key=None):
        if key:
            return self.__sessions.get(key, None)
        else:
            return self.__sessions
    
    def filter(self, **kwargs):
        return self.server.sessions.filter(**kwargs)
    
    def get(self, **kwargs):
        return self.server.sessions.get(**kwargs)
    
    def get_session_with_name(self, session_name, index=0, saveFlag=True):
        if index == -1:
            s = self.server.sessions.filter(session_name=session_name)
        else:
            s = self.server.sessions.filter(session_name=session_name)[index]
        if saveFlag:
            self.__sessions[session_name] = s
        return s

    def create_session(self, name=None):
        name = name if name else sha1(random_string())
        all_sessions_names = self.get_all_sessions_names()
        assert not (name in all_sessions_names), 'name: `{}` is already exist among sessions. plese choose a unique name for session.'.format(name)
        os.system('tmux new-session -d -s "{}"'.format(name))
        return self.get_session_with_name(name)
    
    def __private_get_session_obj(self, session_obj=None):
        sessions_keys = list(self.__sessions.keys())
        if isinstance(session_obj, str):
            if session_obj in sessions_keys:
                return self.__sessions[session_obj]
            elif session_obj in self.get_all_sessions_names():
                _all_sessions = self.get_all_sessions()
                for si in _all_sessions:
                    if si.name == session_obj:
                        return si
            else:
                assert False, 'session=`{}` does not exist.'.format(session_obj)
        if session_obj:
            return session_obj
        elif session_obj is None and len(sessions_keys) == 1:
            return self.__sessions[sessions_keys[0]]
        else:
            assert False, 'session_obj `{}` is not valid.'.format(session_obj)

    def rename_session(self, new_name: str, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.rename_session(str(new_name))
    
    def get_attached_window(self, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.attached_window
    
    def create_window(self, window_name: str, session_obj=None, attach=False):
        window_name = str(window_name)
        assert window_name.lower() != '[tmux]', '`[tmux]` is not valid for window_name'
        session_obj = self.__private_get_session_obj(session_obj)
        session_obj_window_names = [w.name for w in session_obj.windows]
        if window_name in session_obj_window_names:
            assert False, 'window name: `{}` already exist and its better choose unique name!'.format(window_name)
        else:
            return session_obj.new_window(attach=attach, window_name=window_name)
    
    def kill_window(self, window_name: str, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.kill_window(str(window_name))

    def get_window(self, name=None, wid=None, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        if wid == None and name == None:
            if len(session_obj.windows) == 1:
                return session_obj.windows[0]
            else:
                return session_obj.windows
        else:
            if name is not None:
                name = str(name)
                while True:
                    copy_session_obj_windows = session_obj.windows
                    if not ('[tmux]' == str(session_obj.attached_window.name).lower()):
                        break
                    try:
                        session_obj.attached_pane.display_message('This session belongs to the process with PID={}, please don\'t make any changes and stop watching it, any tiny action of you may affect on the flow of the process.'.format(os.getppid()))
                    except Exception as e:
                        pass
                    sleep(1)
                sum_var = sum([1 for wi in copy_session_obj_windows if wi.name == name])
                if sum_var == 1:
                    for w in copy_session_obj_windows:
                        if w.name == name:
                            return w
                elif sum_var == 0:
                    return None # not found
                else:
                    assert False, '(multiple window found) | name: `{}` is not unique in session_name: `{}` and you must search with `window_id` attribute. | windows_list: `{}`'.format(name, session_obj.name, copy_session_obj_windows)
            
            if wid is not None:
                for w in session_obj.windows:
                    if w.id == wid:
                        return w
                return None
            
            return None
    
    def input(self, pane_real_obj, *pargs, **kwargs):
        pane_real_obj.clear()
        while True:
            _temp_out = self.output(pane_real_obj, printFlag=False, ignore_first_and_last_lines=False).split('\n')
            if (len(_temp_out) == 1) and ('@' in _temp_out[0]):
                break
            sleep(1)
        pane_real_obj.send_keys(*pargs, **kwargs)

    def output(self, pane_real_obj, printFlag=False, ignore_first_and_last_lines=True):
        out = '\n'.join(pane_real_obj.cmd('capture-pane', '-p').stdout)
        if ignore_first_and_last_lines:
            out = '\n'.join(out.split('\n')[1 : -1])
        if printFlag:
            print(out)
        return out

    def get_pane(self, window_name=None, window_id=None, window_real_obj=None, pane_id=None, session_obj=None):
        """easy way is just determine window_name"""
        if window_real_obj is not None:
            w = window_real_obj
        else:
            session_obj = self.__private_get_session_obj(session_obj)
            w = self.get_window(name=window_name, wid=window_id, session_obj=session_obj)
        if w is not None:
            if isinstance(w, (list, tuple)):
                assert False, 'w is a list or tuple: `{}` you must exactly determine which w is needed with `window_name` or `window_id` attributes.'.format(w)
            if len(w.panes) == 1:
                return w.panes[0]
            else:
                if pane_id is not None:
                    for pi in w.panes:
                        if pi.id == pane_id:
                            return pi
                    assert False, 'pane_id: `{}` is not exist in window_name: `{}` of session_name: `{}`.'.format(pane_id, w.name, w.session_name)
                else:
                    assert False, '(multiple pane) | there are `{}` panes are exist inside window_name: `{}` of session_name: `{}`. | you can use pain_id to determine pane of pain_list: `{}`.'.format(len(w.panes), w.name, w.session_name, w.panes)
        else:
            assert False, 'window_name: `{}` is not exist in session_name: `{}`'.format(window_name, session_obj.name)

class S3Base:
    """https://alexwlchan.net/2021/s3-progress-bars/"""
    def __init__(self, default_bucket='main', service_provider='LIARA', endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None):
        self.s3 = boto3.resource('s3', 
            endpoint_url= endpoint_url or getenv(f'{service_provider}_ENDPOINT_URL'), 
            aws_access_key_id= aws_access_key_id or getenv(f'{service_provider}_ACCESS_KEY'), 
            aws_secret_access_key= aws_secret_access_key or getenv(f'{service_provider}_SECRET_KEY')
        )
        self.default_bucket = default_bucket
    
    def upload(self, fpath=None, key=None, bucket=None, s3=None, timestampFlag=True):
        key = key if key else os.path.split(fpath)[1]
        if timestampFlag:
            key = key + '_' + getTimeHR_V0()
        s3 = s3 if s3 else self.s3
        bucket = bucket if bucket else self.default_bucket
        file_size = os.stat(fpath).st_size
        with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=fpath) as pbar:
            s3.Bucket(bucket).upload_file(
                Filename=fpath, # file path in this machine.
                Key=key, # filename in the bucket server.
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),)