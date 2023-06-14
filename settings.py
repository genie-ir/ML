import os
from loguru import logger
from os import getenv, environ, makedirs, system, sep
from dotenv import load_dotenv
from libs.basicDS import dotdict
from utils.metrics import Metrics
from os.path import dirname, join, exists
from argparse import ArgumentParser
from libs.basicTime import getTimeHR
from setuptools import find_packages, setup
from libs.basicIO import pathBIO, readBIO, check_logdir, copy_dir

#load env variables.
parser_for_basic_things = ArgumentParser()
parser_for_basic_things.add_argument(
    '--envp',
    type=str,
    default='',
    help='env path',
)
parser_for_basic_things_opt, parser_for_basic_things_unknown = parser_for_basic_things.parse_known_args()
load_dotenv(parser_for_basic_things_opt.envp)
environ['GENIE_ML_REPORT'] = join(os.path.split(parser_for_basic_things_opt.envp)[0], 'report')
makedirs(environ['GENIE_ML_REPORT'], exist_ok=True)

environ['GENIE_ML_STORAGE0'] = environ['GENIE_ML_STORAGE0'].rstrip().rstrip(sep)
assert environ['GENIE_ML_STORAGE0'].endswith(sep + '@GENIE_ML_APP'), 'path `{}` is not valid for expected storage0'.format(environ['GENIE_ML_STORAGE0'])
stg0 = environ['GENIE_ML_STORAGE0']

environ['GENIE_ML_LOGDIR'] = environ['GENIE_ML_LOGDIR'].rstrip().rstrip(sep)
assert environ['GENIE_ML_LOGDIR'].endswith(sep + 'logs'), '`GENIE_ML_LOGDIR` it must be ends with `logs` but now is: {}'.format(environ['GENIE_ML_LOGDIR'])

# setup the project
if getenv('SETUP') == 'True':
    setup(
        name=getenv('NAME'),
        packages=find_packages(),
        version=getenv('VERSION'),
        description=getenv('DESCRIPTION'),
        author=getenv('AUTHOR'),
        license=getenv('LICENSE'),
    )

# load some of ARGS
parser = ArgumentParser()
parser.add_argument(
    '--main',
    type=str,
    default='app',
    help='main functionality | default is `app`',
)
parser.add_argument(
    '--app',
    type=str,
    default='.default', # starts with dot -> hidden folder
    help='app name',
)
parser.add_argument(
    '--app-fn',
    type=str,
    default='main',
    help='app function name',
)
opt, unknown = parser.parse_known_args()
opt_main = opt.main

if opt_main == 'app' and opt.app == '.default':
    assert False, '`app` is required'

app_splited = [str(s).strip() for s in opt.app.split(':')]
environ['GENIE_ML_APP_MD'] = app_splited[0]

app_splited_status = None
if len(app_splited) == 1 and len(app_splited[0]) > 0:
    environ['GENIE_ML_APP'] = app_splited[0]
elif len(app_splited) == 2 and len(app_splited[0]) > 0 and len(app_splited[1]) > 0:
    environ['GENIE_ML_APP'] = app_splited[1]
    app_splited_status = 'CODE0'
elif len(app_splited) == 3 and len(app_splited[0]) > 0 and len(app_splited[1]) > 0 and len(app_splited[2]) > 0:
    environ['GENIE_ML_APP'] = app_splited[2]
    app_splited_status = 'CODE1'
else:
    raise ValueError('app name `{}` is not valid. it should be like (appname | appname:newapp | appname:oldapp:newapp)'.format(opt.app))

environ['GENIE_ML_APP_FN'] = opt.app_fn

PFX_KEYS = environ['GENIE_ML_PFX'].split(',')

for k, v in environ.items():
    flag = False
    for valid_prefix in PFX_KEYS:
        if k.startswith(valid_prefix):
            flag = True
            break
    
    if not flag:
        continue
    
    new_v = []
    for vi in v.split(sep):
        if vi.startswith('@'):
            new_v.append(environ[vi[1:]])
        else:
            new_v.append(vi)
    new_v = sep.join(new_v)
    environ[k] = new_v

src_dir = None

if app_splited_status == 'CODE0' and (not exists(environ['GENIE_ML_STORAGE0'])):
    src_dir = join(os.path.split(environ['GENIE_ML_STORAGE0'])[0], app_splited[0])
    if exists(src_dir):
        copy_dir(src_dir, environ['GENIE_ML_STORAGE0'], waitFlag=True)
    else:
        pass # Not Statement

if app_splited_status == 'CODE1' and (not exists(environ['GENIE_ML_STORAGE0'])):
    src_dir = join(os.path.split(environ['GENIE_ML_STORAGE0'])[0], app_splited[1])
    if exists(src_dir):
        copy_dir(src_dir, environ['GENIE_ML_STORAGE0'], waitFlag=True)
    else:
        raise ValueError('src dir `{}` is not exist'.format(src_dir))

src_dir = None

# https://github.com/Kaggle/kaggle-api
kaggledict = readBIO(join(str(getenv('KAGGLE_CONFIG_DIR')), 'kaggle.json'), dotdictFlag=False)
kaggledict['key'] = kaggledict.get('key', '') or getenv('KAGGLE_KEY')
kaggledict['username'] = kaggledict.get('username', '') or getenv('KAGGLE_USERNAME')

if getenv('KAGGLE_CHMOD'):
    system('chmod {} {}'.format(
        str(getenv('KAGGLE_CHMOD')),
        join(str(getenv('KAGGLE_CONFIG_DIR')), 'kaggle.json')
    ))

os.environ['KAGGLE_KEY'] = kaggledict['key']
os.environ['KAGGLE_USERNAME'] = kaggledict['username']

import kaggle # need to import here(after env variables had defined)