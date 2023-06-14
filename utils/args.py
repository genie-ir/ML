import os
from loguru import logger
from libs.coding import sha1
from omegaconf import OmegaConf
from os import getenv, environ , makedirs
from libs.basicIO import ls, pathBIO, merge_files
from pytorch_lightning import seed_everything
from os.path import join, exists, isfile, isdir
from argparse import ArgumentParser, ArgumentTypeError
from abc import ABC, abstractmethod
from pytorch_lightning.trainer import Trainer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


class ParserBasic(ABC):
    def __new__(cls, **kwargs):
        super().__new__(cls)
        return cls.parser(**kwargs)
    
    @classmethod
    def nondefault_trainer_args(cls, opt):
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args([]) # this is specefic syntax and its mean only return know params[Trainer params] with default values.
        return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

    @classmethod
    def predefined_args(cls, parser):
        parser.add_argument(
            '--simlink-rootdir',
            type=str,
            default='',
            help='Do not set it manualy | create a simlink to GENIE_ML_ROOT',
        )
        parser.add_argument(
            '--simlink-storagei',
            type=str,
            default='',
            help='Do not set it manualy | create a simlink to GENIE_ML_STORAGE0',
        )
        parser.add_argument(
            '--app',
            type=str,
            help='app name',
        )
        parser.add_argument(
            '--Rfn',
            type=str,
            default='',
            help='Rfn | Optinal (If this provid then It be overwrite on coresponding config yaml file)',
        )
        parser.add_argument(
            '-L',
            '--logger_ml',
            type=str,
            default='genie', #'tensorboard',
            help='default_logger_cfgs key name',
        )
        parser.add_argument(
            '-M',
            '--metrics_tbl',
            type=str,
            const=True,
            default=None,
            nargs='?',
            help='metrics table name',
        )
        parser.add_argument(
            '-C',
            '--ckpt-fname',
            '--ckpt',
            type=str,
            default='last',
            help='ckpt fname (default=last) | consider absolute path if ends with `.ckpt` or relative ckpt file in checkpoints location. | it can be a directory path in case of checkpoint spliting',
        )
        parser.add_argument(
            '-H',
            '--hash-ignore',
            nargs='*',
            help='hash ignore list for plLogger(Geine)',
            default=[''],
        )
        parser.add_argument(
            '-n',
            '--name',
            type=str,
            const=True,
            default='',
            nargs='?',
            help='postfix for logdir',
        )
        parser.add_argument(
            '-r',
            '--resume',
            type=str,
            const=True,
            default='',
            nargs='?',
            help='resume from logdir | it must be a directory path if it passed',
        )
        parser.add_argument(
            '-b',
            '--base',
            nargs='*',
            metavar='base_config.yaml',
            help='paths to base configs. Loaded from left-to-right. '
            'Parameters can be overwritten or added with command-line options of the form `--key value`.',
            default=list(),
        )
        parser.add_argument(
            '-t',
            '--train',
            type=str2bool,
            const=True,
            default=False,
            nargs='?',
            help='train',
        )
        parser.add_argument(
            '--no-test',
            type=str2bool,
            const=True,
            default=False,
            nargs='?',
            help='disable test',
        )
        parser.add_argument(
            '--no-validate',
            type=str2bool,
            const=True,
            default=False,
            nargs='?',
            help='disable validate',
        )
        parser.add_argument('-p', '--project', help='name of new or path to existing project')
        parser.add_argument(
            '-d',
            '--debug',
            type=str2bool,
            nargs='?',
            const=True,
            default=False,
            help='enable post-mortem debugging',
        )
        parser.add_argument(
            '-s',
            '--seed',
            type=int,
            default=-1,
            help='seed for seed_everything | default value is -1 for dont use of seed you can use 23 or anything for considering seed value',
        )
        parser.add_argument(
            '-f',
            '--postfix',
            type=str,
            default='',
            help='post-postfix for default name',
        )

        return parser

    @classmethod
    def parser(cls, **kwargs):
        ctlFlag = kwargs.get('ctlFlag', True)
        predefFlag = kwargs.get('predefFlag', True)
        trainerFlag = kwargs.get('trainerFlag', True)

        parser = cls.get_parser(**kwargs)
        if predefFlag:
            parser = cls.predefined_args(parser)
        if trainerFlag:
            parser = Trainer.add_argparse_args(parser)
        opt, unknown = parser.parse_known_args()
        
        if ctlFlag:
            return opt, unknown, cls.ctl_parser(opt, unknown, **kwargs)
        return opt, unknown
    
    @classmethod
    @abstractmethod
    def get_parser(cls, **kwargs):
        parser = ArgumentParser(**kwargs.get('getParserFn_kwargs', dict()))
        return parser
    
    @classmethod
    def ctl_parser(cls, opt, unknown, **kwargs):
        now = kwargs['now']

        if bool(opt.simlink_rootdir):
            os.system('ln -sf {} {}'.format(getenv('GENIE_ML_ROOT'), opt.simlink_rootdir))
        
        if bool(opt.simlink_storagei):
            environ_keys = list(environ.keys())
            for environ_keys_i in environ_keys:
                if (environ_keys_i).lower().startswith('GENIE_ML_STORAGE'.lower()):
                    os.system('ln -sf {} {}'.format(getenv(environ_keys_i), opt.simlink_storagei + environ_keys_i[len('GENIE_ML_STORAGE'):]))

        if opt.name and opt.resume:
            raise ValueError(
                '-n/--name and -r/--resume cannot be specified both.'
                'If you want to resume training in a new log folder, '
                'use -n/--name in combination with --resume_from_checkpoint'
            )
        if opt.resume:
            if isdir(opt.ckpt_fname):
                src_of_opt_ckpt_fname = opt.ckpt_fname # directory contains of splited checkpoint.
                dst_of_opt_ckpt_fname = join(getenv('GENIE_ML_CACHEDIR'), '{}__merged'.format(sha1(opt.ckpt_fname))) # file address of merging process (this is a zip file usually)
                dst_unzipped_dir = merge_files(src=src_of_opt_ckpt_fname, dst=dst_of_opt_ckpt_fname, waitFlag=True, extraction_mode='zip') # unziped directory
                dst_unzipped_dir_items = ls(dst_unzipped_dir, '*.ckpt', full_path=True) # (real ckpt file) inside unziped directory
                logger.critical(dst_unzipped_dir_items)
                assert isinstance(dst_unzipped_dir_items, list) and len(dst_unzipped_dir_items) == 1 and str(dst_unzipped_dir_items[0]).endswith('.ckpt'), 'dst_unzipped_dir_items is not valid it must be have `exact one item` with `.ckpt` extention but now is -> {}'.format(dst_unzipped_dir_items)
                opt.ckpt_fname = dst_unzipped_dir_items[0] # must be ends with `.ckpt` becuse it considred as absolute path.

            assert opt.resume != '@', 'opt.resume=`{}` is not valid'.format(opt.resume)
            if str(opt.resume).startswith('@'):
                opt.resume = join(getenv('GENIE_ML_LOGDIR'), opt.resume[1:])
            
            # if not exists(opt.resume):
            #     makedirs(opt.resume, exist_ok=True)
            
            if isfile(opt.resume): # ckpt address
                raise ValueError('opt.resume must be refer to `logdir` but now refer to a file. | opt.resume={}'.format(opt.resume))
            else: # logdir address
                assert isdir(opt.resume), '{} is must be directory'.format(opt.resume)
                logdir = opt.resume.rstrip('/')
            
            if str(opt.ckpt_fname).endswith('.ckpt'): # absolute path
                ckpt = opt.ckpt_fname
            else:
                ckpt = join(getenv('GENIE_ML_CKPTDIR') or join(logdir, 'checkpoints'), opt.ckpt_fname + '.ckpt')

            assert exists(ckpt), 'ckpt path `{}` does not exist.'.format(ckpt)
            opt.resume_from_checkpoint = ckpt
            base_configs = sorted(
                ls(logdir, 'configs/*.yaml', full_path=True)
            )


            aa = ls(logdir, 'configs/*.yaml', full_path=True)
            bb = sorted(aa)
            print('*'*30)
            print('aa', aa)
            print('bb', bb)

            opt.base = base_configs + opt.base
            _tmp = logdir.split('/')
            nowname = _tmp[_tmp.index('logs')+1]
        else:
            if opt.name:
                name = '_' + opt.name
            elif opt.base:
                cfg_fname = os.path.split(opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = '_' + cfg_name
            else:
                name = ''
            nowname = now + name + opt.postfix
            logdir = join(pathBIO(getenv('GENIE_ML_LOGDIR')), nowname)

        ckptdir = getenv('GENIE_ML_CKPTDIR') or join(logdir, 'checkpoints')
        cfgdir = getenv('GENIE_ML_CFGDIR') or join(logdir, 'configs')
        
        if int(opt.seed) != -1:
            seed_everything(opt.seed)
        return ckptdir, cfgdir, logdir, nowname
