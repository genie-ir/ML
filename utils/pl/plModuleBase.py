import os
import sys
import torch
import cowsay
import numpy as np
import torch.nn as nn
from loguru import logger
from tabulate import tabulate
import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchsummary import summary
from stringcase import pascalcase
from utils.pl.plPlot import fwd_plot
from libs.dyimport import instantiate_from_config
from libs.basicIO import signal_save, compressor, puml, fwrite

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

class plModuleBase(pl.LightningModule):
    def __init__(self,
        pipconfig=None,
        netconfig=None,
        optconfig=None,
        lossconfig=None,
        **kwargs
    ):
        super().__init__()

        for karg, varg in kwargs.items():
            setattr(self, karg, varg)
        
        self.signal_key = str(kwargs.get('signal_key', 'X'))

        self.pipconfig = OmegaConf.to_container(pipconfig or OmegaConf.create())
        self.netconfig = OmegaConf.to_container(netconfig or OmegaConf.create())
        self.optconfig = OmegaConf.to_container(optconfig or OmegaConf.create())
        self.lossconfig = OmegaConf.to_container(lossconfig or OmegaConf.create())
        self.automatic_optimization = False
        
        ##############################[network&optimizer configuration]##################################
        if len(self.netconfig) == 0:
            self.netconfig['lab'] = {'target': '.ignore'}
            __rfn = str(kwargs.get('Rfn', ''))
            if __rfn:
                if __rfn.endswith('lab'):
                    kwargs['Rfn'] = __rfn
                else:
                    assert False, '`Rfn={}` | It should be ends with `lab`.'.format(__rfn)
            else:
                kwargs['Rfn'] = '_lab'
            setattr(self, 'Rfn', kwargs['Rfn'])
        __optconf_map = dict()
        for default_index, (neti_name, neti) in enumerate(self.netconfig.items()):
            self.netconfig[neti_name]['metrics'] = neti.get('metrics', ['loss'])
            self.netconfig[neti_name]['optimizer'] = neti.get('optimizer', dict())
            self.netconfig[neti_name]['optimizer']['index'] = int(default_index) # indexies only assign by code not by user.
            self.netconfig[neti_name]['optimizer']['target'] = str(neti['optimizer'].get('target', 'torch.optim.Adam')) + '.dont'
            self.netconfig[neti_name]['optimizer']['params'] = neti['optimizer'].get('params', dict())
            __optconf_map[self.netconfig[neti_name]['optimizer']['index']] = str(neti_name)
        self.optconfig['map'] = __optconf_map # mapping only done by code not by user
        self.optconfig['batch_learning_frequency'] = self.optconfig.get('batch_learning_frequency', ','.join(['{}:{}'.format(str(opti), str(int(self.netconfig[netn]['optimizer'].get('blf', 1)))) for opti, netn in self.optconfig['map'].items()]))
        self.optconfig['epoch_learning_frequency'] = self.optconfig.get('epoch_learning_frequency', {int(opti): int(self.netconfig[netn]['optimizer'].get('elf', 1)) for opti, netn in self.optconfig['map'].items()})
        ##################################################################################################
        
        optimizationIndex = self.optconfig['batch_learning_frequency']
        if isinstance(optimizationIndex, str):
            _opt_idx_list = []
            for oi in (optimizationIndex.split(',')):
                oi_split = oi.split(':')
                assert len(oi_split) == 2, '`optimizationIndex={}` is not valid'.format(optimizationIndex)
                oil_index, oir_value = oi_split
                _opt_idx_list = _opt_idx_list + [int(oil_index)] * int(oir_value)
            self.optimizationIndex = _opt_idx_list
        else:
            assert isinstance(optimizationIndex, (list, tuple)), 'type(optimizationIndex)={} | is not valid.'.format(type(optimizationIndex))
            self.optimizationIndex = list(optimizationIndex)
        assert len(self.optimizationIndex) > 0, 'optimizationIndex is empty. | It must be defined at particular config file.'
            
        self.networks = dict()
        self.lossconfig['target'] = str(self.lossconfig.get('target', 'apps.{}.modules.losses.loss.Loss'.format(self.__class__.__name__)))
        self.lossconfig['params'] = self.lossconfig.get('params', dict())

        visualize_vars = [{}, {}]         # first: for network | second: for pipline
        visualize_flags = [False, False]  # first: for network | second: for pipline
        assert len(visualize_vars) == 2 and len(visualize_flags) == 2
        visualize_fn = kwargs.get('visualize', 'puml')
        
        if visualize_fn == 'puml':
            for vidx in range(2):
                visualize_vars[vidx] = {
                    'puml_R': [],
                    'puml_N': []
                }
            
        for netcfg_k, netcfg_v  in self.netconfig.items():
            if not ('target' in netcfg_v):
                netcfg_v['target'] = 'utils.pt.nnModuleBase.nnModuleBase'
                self.netconfig[netcfg_k]['target'] = netcfg_v['target']
            netcfg_v['params'] = netcfg_v.get('params', dict())
            assert isinstance(netcfg_v['params'], dict), '`type(netcfg_v["params"])={}` | It must be `dict`'.format(type(netcfg_v['params']))
            netcfg_v['params']['_chain_name'] = pascalcase(self.__class__.__name__) + '.' + pascalcase(netcfg_k)
            netcfg_v['params']['n_net'] = len(self.netconfig)
            
            self.networks[netcfg_k] = instantiate_from_config(netcfg_v, kwargs=kwargs)
            setattr(self, netcfg_k, self.networks[netcfg_k])
            self.lossconfig['params']['netlossfn'] = f'{netcfg_k}_loss'
            setattr(self, f'{netcfg_k}Loss', instantiate_from_config(self.lossconfig))

            _net_step_function = self.predefined_net_step_master0(netcfg_k)
            netpip = self.pipconfig.get(f'{netcfg_k}_step', None)
            
            if (not (netpip is None)) and (isinstance(netpip, dict)):
                netpip['target'] = str(netpip.get('target', 'utils.pt.nnModuleBase.nnModuleBase'))
                target_netpip = netpip['target']
                del netpip['target']
                net_pipline = instantiate_from_config({
                    'target': target_netpip,
                    'params': {
                        **netpip,
                        'STEPWISE': True,
                        '_chain_name': pascalcase(self.__class__.__name__) + '.' + pascalcase(self.net2pipline(netcfg_k)),
                        'n_net': len(self.pipconfig)
                    }
                }, kwargs=kwargs)
                setattr(self, self.net2pipline(netcfg_k), net_pipline)
                _net_step_function = getattr(self, 'predefined_net_step_master')(netcfg_k)

            setattr(self, f'_{netcfg_k}_step', _net_step_function)
            if getattr(self, f'{netcfg_k}_step', None) is None:
                setattr(self, f'{netcfg_k}_step', getattr(self, f'_{netcfg_k}_step'))

            for v_idx, visualize_part in enumerate([netcfg_k, self.net2pipline(netcfg_k)]):
                visualize_network = getattr(self, visualize_part, None)
                if visualize_network:
                    visualize_output = getattr(visualize_network, 'visualize', lambda *pa, **kw: {})(visualize_fn=visualize_fn)
                
                    if visualize_fn == 'puml' and isinstance(visualize_output, dict):
                        visualize_flags[v_idx] = True
                        visualize_vars[v_idx]['puml_R'] = visualize_vars[v_idx]['puml_R'] + visualize_output.get('R', [])
                        visualize_vars[v_idx]['puml_N'] = visualize_vars[v_idx]['puml_N'] + visualize_output.get('N', [])

        assert len(self.networks) > 0, 'self.networks is empty'

        v_idx_mapname = ['model', 'pipline']
        for v_idx in range(2):
            if visualize_flags[v_idx]:
                if visualize_fn == 'puml':
                    puml_sudocode = visualize_vars[v_idx]['puml_R'] + visualize_vars[v_idx]['puml_N']
                    if len(puml_sudocode) > 0:
                        puml(f'.{v_idx_mapname[v_idx]}.puml', f'{v_idx_mapname[v_idx]}.png', u='\n'.join(puml_sudocode))
        
        
        
        tabulate_list = {'model': [], 'pipline': []}
        for net_k, net_v in self.networks.items():
            tabulate_list['model'].append([net_k, net_v])
            net_k_coresponding_pipline = self.net2pipline(net_k)
            net_coresponding_pipline = getattr(self, net_k_coresponding_pipline, None)
            if net_coresponding_pipline:
                tabulate_list['pipline'].append([net_k_coresponding_pipline, net_coresponding_pipline])

        
        for v_idx in range(2):    
            tbl = tabulate(tabulate_list[v_idx_mapname[v_idx]], tablefmt='grid', headers=['Network', 'Architecture'])
            fwrite(os.path.join(os.getenv('GENIE_ML_REPORT'), f'{v_idx_mapname[v_idx]}.txt'), tbl)
            if kwargs.get('print', False):
                print(tbl)


        self.empty_log_dict = dict()
        for eld_split in list(kwargs.get('eld_split_list', ['train', 'val', 'test'])):
            self.empty_log_dict[eld_split] = dict()
            for network_name in list(self.netconfig.keys()):
                for metric_name in self.netconfig[network_name]['metrics']:
                    self.empty_log_dict[eld_split]['{}/{}_{}'.format(eld_split, network_name, metric_name)] = float('nan')
        
        self.Rfn = str(kwargs.get('Rfn', '')) # Notic: [empty string -> Nothing happend] becuse it casted as `False`
        if bool(self.Rfn):
            if not self.Rfn.startswith('_'):
                self.Rfn = '_' + self.Rfn

            if self.Rfn.endswith('lab') or self.Rfn in ['_synthesis', '_plot']:
                for specificfn in ['forward', 'training_step', 'validation_step', 'on_validation_epoch_end']:
                    setattr(
                        self,
                        specificfn + self.Rfn, 
                        getattr(self, specificfn + self.Rfn, getattr(self, f'predefined_{specificfn}'))
                    )

            rfn_list = [elementName for elementName in dir(self) if elementName.endswith(self.Rfn)]
            rfn_list = list(dict.fromkeys(rfn_list).keys())
            RfnLen = -len(self.Rfn) # Negetive number
            
            for fnName in rfn_list:
                setattr(self, 'legacy_' + fnName[:RfnLen], getattr(self, fnName[:RfnLen], lambda *parg, **karg: None))
                setattr(self, fnName[:RfnLen], getattr(self, fnName))

        self.ignore_keys = list(kwargs.get('ignore_keys', []))
        self.ckpt_path = str(kwargs.get('ckpt_path', ''))
        if bool(self.ckpt_path):
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)

    def net2pipline(self, netname):
        return f'{netname}Step'
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        """It can be overwrite in child class"""
        logger.critical(f'Restored from {path}')
        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
    
    def predefined_net_step_master0(self, net):
        pipline_name = f'{net}_step'
        def predefined_net_step(batch):
            cowsay.cow('NotImplementedError:\nplease define pipline `{}`.'.format(pipline_name))
            sys.exit()
        return predefined_net_step
    
    def predefined_net_step_master(self, net):
        pipline_name = self.net2pipline(net)
        def predefined_net_step_slave(batch):
            return getattr(self, pipline_name).forward(**batch)
        return predefined_net_step_slave
    
    def getbatch(self, batch):
        """It can be overwrite in child class"""
        batch['batch_size'] = batch[self.signal_key].shape[0]
        batch['device'] = self.device
        batch['Self'] = self
        return batch
    
    def training_step(self, batch, batch_idx, split='train'):
        """It can be overwrite in child class"""
        batch = self.getbatch(batch)
        optimizers_list = self.optimizers()
        if not isinstance(optimizers_list, (list, tuple)):
            optimizers_list = [optimizers_list]
        
        log_dict = {**self.empty_log_dict[split]}
        for optimizer_idx in self.optimizationIndex:
            if (self.current_epoch+1) % self.optconfig['epoch_learning_frequency'].get(optimizer_idx, 1) != 0:
                continue
            
            optimizers_list[optimizer_idx].zero_grad()
            cnet = self.optconfig['map'][optimizer_idx] # current network
            loss, _ld = getattr(self, '{}_step'.format(cnet))(batch)
            ld = dict(('{}/{}_{}'.format(split, cnet, cnet_metric), _ld[cnet_metric]) for cnet_metric in self.netconfig[cnet]['metrics'])
            log_dict = {**log_dict, **ld}
            # optimizers_list[optimizer_idx].zero_grad()
            self.manual_backward(loss)
            optimizers_list[optimizer_idx].step()
        
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batch['batch_size'])
    
    def validation_step(self, batch, batch_idx, split='val'):
        """It can be overwrite in child class"""
        batch = self.getbatch(batch)

        log_dict = {**self.empty_log_dict[split]}
        for cnet in self.netconfig:
            loss, _ld = getattr(self, '{}_step'.format(cnet))(batch)
            ld = dict(('{}/{}_{}'.format(split, cnet, cnet_metric), _ld[cnet_metric]) for cnet_metric in self.netconfig[cnet]['metrics'])
            log_dict = {**log_dict, **ld}

        # DONT CHANGE BELOW FLAGS
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch['batch_size'])

    def forward(self, *args, **kwargs):
        print('orginal forward')
        raise NotImplementedError()
    
    
    ##################[only once at end of val epoch]#################### [default]
    def predefined_training_step(self, batch, batch_idx, split='train'):
        return

    def predefined_validation_step(self, batch, batch_idx, split='val'):
        return
    
    def predefined_forward(self, *args, **kwargs):
        logger.error('NotImplementedError: `predefined_forward`')
        return NotImplementedError()

    def predefined_on_validation_epoch_end(self):
        res = self()
        if isinstance(res, NotImplementedError):
            cowsay.cow('NotImplementedError:\nplease define `{}` function.'.format('forward' + self.Rfn))
        else:
            cowsay.cow('done')
        sys.exit()
    ######################################################################

    ##################[only once at end of val epoch]#################### [overwrite]
    def forward_synthesis(self): # only once at end of val epoch
        dirname = getattr(self, 'synthesis_dirname', 'synthesis')
        for c in range(getattr(self, 'nclass', 1)): # TODO
            N = int(getattr(self, 'nsynthesis', dict()).get(c, 200))
            for n in range(N):
                B = 1
                y = torch.zeros((B,), device=self.device).long() + c
                generated_signal = self.legacy_forward(B=B, y=y).argmax(dim=1, keepdim=True).squeeze()
                spath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), dirname, f'class_{c}', f'{n}.npy')
                signal_save(generated_signal, spath)
        dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), dirname)
        compressor(dpath, f'{dpath}.zip', mode='zip')
    ######################################################################

    ##################[every step - one period]#################### [plot]
    def forward_plot(self, batch):
        return fwd_plot(batch=batch, plot_params=getattr(self, 'plot', dict()), sigkey=self.signal_key)
                   
    def training_step_plot(self, batch, batch_idx, split='train'):
        self.training_step_flag = True
        return self(batch)
    
    def validation_step_plot(self, batch, batch_idx, split='val'):
        if bool(getattr(self, 'training_step_flag', False)):
            return self(batch)
    
    def on_validation_epoch_end_plot(self):
        if bool(getattr(self, 'training_step_flag', False)):
            cowsay.cow('done')
            sys.exit()
    ################################################################

    
    
    # def training_step_lab(self, batch, batch_idx, split='train'):
    #     return
    
    # def validation_step_lab(self, batch, batch_idx, split='val'):
    #     print('ok!', batch_idx)
    #     return
    
    # def on_validation_epoch_end_lab(self):
    #     self._lab()
    #     assert False, 'END'

    def get_pretrained_model(self, config=None, model=None, freezeFlag=True):
        if model is None and config is not None:
            model = instantiate_from_config(config)
        if freezeFlag:
            # model = model.eval()
            # model.train = disabled_train
            model.requires_grad_(False)
        return model
    
    def log_signal(self, batch, **kwargs):
        """
            It can be overwrite in child class
            Notic: this function is must be exist in ptCallback.ImageLoggerBase
        """
        print('log_signal')
        assert False
        log = dict()
        X = self.get_input(batch, self.signal_key).to(self.device)
        Y, _ = self(X)
        log['X'] = X # log['inputs'] = x
        log['Y'] = Y
        return log
    
    def configure_optimizers(self):
        __real_scl_list = []
        __real_opt_list = []
        for opt_i in range(len(self.optconfig['map'])):
            net_name = self.optconfig['map'][int(opt_i)]
            if getattr(self, net_name) is None:
                __real_opt_list.append(None)
            else:
                __real_opt_list.append(instantiate_from_config(self.netconfig[net_name]['optimizer'])(getattr(self, net_name).parameters(), **self.netconfig[net_name]['optimizer']['params']))
        if len(__real_scl_list) == 0 and len(__real_opt_list) == 1 and __real_opt_list[0] is None:
            return None
        
        return __real_opt_list, __real_scl_list