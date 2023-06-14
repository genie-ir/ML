import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import os, re, sys, cowsay
from libs.basicIO import puml
import torch.nn.functional as F
from libs.basicDS import dotdict
from stringcase import pascalcase
from libs.coding import random_string
from utils.pt.building_block import BB
from utils.pt.BB.Join import Join as BB_Join
from utils.pt.BB.Self import Self as BB_Self
from libs.dyimport import instantiate_from_config
from utils.pt.loader import *

class DynamicActive_nnModule(nn.Module):
    def __init__(self, fn_name, fn_params=None):
        super().__init__()
        
        fn_params = dict() if fn_params is None else fn_params
        
        self.fn_name = fn_name
        self.fn_params = fn_params
        
        if isinstance(self.fn_params, dict):
            self.unpackstar = '**'
        elif isinstance(self.fn_params, (list, tuple)):
            self.unpackstar = '*'
            assert False, '`__DynamicActive_nnModule` only supports `fn_params` as `dict` | fn_name={} | fn_params={}'.format(fn_name, fn_params)
        else:
            assert False, '`type(fn_params)={}` | It must be `dict` or `list or tuple` | fn_name={} | fn_params={}'.format(type(fn_params), fn_name, fn_params)

        if len(self.fn_params) == 0:
            self.bypassCode = ''
        else:
            self.bypassCode = f', {self.unpackstar}self.fn_params'

    def forward(self, *args, **batch):
        return eval(f'{self.fn_name}(*args, **batch{self.bypassCode})')

class DynamicPassive_nnModule(nn.Module):
    def __init__(self, fn_name, fn_params=None):
        super().__init__()
        
        fn_params = dict() if fn_params is None else fn_params
        
        self.fn_name = fn_name
        self.fn_params = fn_params
        
        if isinstance(self.fn_params, dict):
            self.unpackstar = '**'
        elif isinstance(self.fn_params, (list, tuple)):
            self.unpackstar = '*'
        else:
            assert False, '`type(fn_params)={}` | It must be `dict` or `list or tuple` | fn_name={} | fn_params={}'.format(type(fn_params), fn_name, fn_params)

        if len(self.fn_params) == 0:
            self.bypassCode = ''
        else:
            self.bypassCode = f'{self.unpackstar}self.fn_params'

    def forward(self, x):
        return eval(f'x.{self.fn_name}({self.bypassCode})')

class nnModuleBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.start()
    
    def start(self):
        self.kwargs['STEPWISE'] = bool(self.kwargs.get('STEPWISE', False))
        self.kwargs['_chain_name'] = str(self.kwargs.get('_chain_name', ''))
        self.kwargs['nclass'] = int(self.kwargs.get('nclass', -1))
        self.reparameterization()
        self.kwargs_dotdict = dotdict(self.kwargs)
        self.forward_list = list(self.kwargs.get('forward', []))

        for network_topology in ['seq', 'tree', 'dag', 'graph']:
            self.networks_builder(str(network_topology).lower())
        
        if bool(self.kwargs.get('apply_weights_init', True)):
            self.apply(getattr(self, self.kwargs.get('weights_init_fn_name', 'weights_init')))

    def __network_builder_seq(self, part: str, ncfg: dict, N=None):
        N = int(2 if N is None else N)
        list_vars = dict()
        for ncfg_k, ncfg_v in ncfg.items():
            if isinstance(ncfg_v, (list, tuple)):
                flag = True
                for ncfg_v_i in ncfg_v:
                    flag = flag and isinstance(ncfg_v_i, (int, float))
                if flag:
                    list_vars[ncfg_k] = ncfg_v
        list_vars = dotdict(list_vars)
        
        netlayers = []
        for i in range(N - 1):
            for _node in ncfg[part]:
                if (isinstance(_node, str) and _node.startswith('.')) or (isinstance(_node, dict) and len(_node) > 0 and list(_node.keys())[0].startswith('.')):
                    if isinstance(_node, str):
                        if _node.startswith('..'):
                            _node_ins = type(pascalcase(_node[2:].replace('.', '_')), (DynamicActive_nnModule,), {})(fn_name=_node[2:])
                        else:
                            _node_ins = type(pascalcase(_node[1:].replace('.', '_')), (DynamicPassive_nnModule,), {})(fn_name=_node[1:])
                    elif isinstance(_node, dict):
                        assert len(_node) == 1 , '`len(_node)={}` | It must be `1` | looking this maybe useful: `_node={}`'.format(len(_node), _node)
                        _node_key = list(_node.keys())[0]

                        if isinstance(_node[_node_key], dict):
                            _tempval = dict()
                            for _tempk, _tempv in _node[_node_key].items():
                                if isinstance(_tempv, str) and '$' in _tempv:
                                    _tempval[_tempk] = eval(_tempv.replace('$', 'self.kwargs_dotdict.'))
                                else:
                                    _tempval[_tempk] = _tempv
                        elif isinstance(_node[_node_key], (list, tuple)):
                            _tempval = []
                            for _tempv in _node[_node_key]:
                                if isinstance(_tempv, str) and '$' in _tempv:
                                    _tempval.append(eval(_tempv.replace('$', 'self.kwargs_dotdict.')))
                                else:
                                    _tempval.append(_tempv)
                        else:
                            _tempval = _node[_node_key]

                        if _node_key.startswith('..'):
                            _node_ins = type(pascalcase(_node_key[2:].replace('.', '_')), (DynamicActive_nnModule,), {})(fn_name=_node_key[2:], fn_params=_tempval)
                        else:
                            _node_ins = type(pascalcase(_node_key[1:].replace('.', '_')), (DynamicPassive_nnModule,), {})(fn_name=_node_key[1:], fn_params=_tempval)
                    else:
                        assert False, '`type(_node)={}` | It must be `str` or `dict` | maybe looking it be useful: `ncfg={}`'.format(type(_node), ncfg)
                    
                    netlayers.append(_node_ins)
                else:
                    node = _node
                    if isinstance(node, str) and node.startswith('BB.'):
                        node = 'utils.pt.' + node + '.' + os.path.split(node.replace('.', '/'))[1]
                    elif isinstance(node, dict) and len(node) == 1 and list(node.keys())[0].startswith('BB.'):
                        nodek = list(node.keys())[0]
                        nodev = node[nodek]
                        nodek_new = 'utils.pt.' + nodek + '.' + os.path.split(nodek.replace('.', '/'))[1]
                        nodev_new = dict()
                        
                        if isinstance(nodev, dict):
                            for nodev_key, nodev_value in nodev.items():
                                if isinstance(nodev_value, dict):
                                    for nodev_value_i_key, nodev_value_i_value in nodev_value.items():
                                        nodev_new['{}:{}'.format(nodev_key, nodev_value_i_key)] = nodev_value_i_value
                                else:
                                    nodev_new[nodev_key] = nodev_value
                        else:
                            nodev_new = nodev
                        
                        node = {nodek_new: nodev_new}
                    else:
                        pass # Not Statement

                    if isinstance(node, str):
                        netlayers.append(instantiate_from_config({
                            'target': node,
                            'params': {}
                        }))
                    elif isinstance(node, dict):
                        for _target, __params in node.items():
                            _params = dict()
                            for _k, _v in __params.items():
                                if isinstance(_v, str) and '$' in _v:
                                    _params[_k] = eval(_v.replace('$', 'self.kwargs_dotdict.'))
                                else:
                                    _params[_k] = _v
                            if _params.get('kernel_size', 0) == -1:
                                continue
                            netlayers.append(instantiate_from_config({
                                'target': _target,
                                'params': _params
                            }))
                    else:
                        assert False, '`type(node)={}` | each node it must be `str` or `dict` | looking this maybe useful: `ncfg={}`'.format(type(node), ncfg)
        return netlayers

    def network_builder_seq(self, ncfg: dict, part_list=None, N_dict=None):
        N_dict = dict() if N_dict is None else N_dict
        
        inp = ncfg.get('input', None)
        out = ncfg.get('output', None)
        assert ((inp is None) or (isinstance(inp, str) and len(inp) > 0)) , '`input={}` | It must be `str` with positive length | (topology=seq)'.format(inp)
        assert isinstance(out, str) and len(out) > 0 , '`output={}` | It must be `str` with positive length | (topology=seq)'.format(out)

        part_list = ['head', 'supernode', 'tail'] if part_list is None else part_list
        assert isinstance(part_list, (list, tuple)), '`type(part_list)={}` | It must be `list` or `tuple` of `strings`'.format(type(part_list))
        if not ('f' in ncfg):
            ncfg['f'] = []
        assert isinstance(ncfg['f'], (list, tuple)), '`type(ncfg["f"])={}` | It must be `list` or `tuple` of `numbers`'.format(type(ncfg['f']))
        
        netlayers = []
        for part in list(part_list):
            assert isinstance(part, str), '`type(part)={}`| each element of `part_list` should be a `str`'.format(type(part))
            ncfg[part] = ncfg.get(part, None)
            
            if ncfg[part] is None:
                continue
            assert isinstance(ncfg[part], (list, tuple)), '`type(ncfg[part])={}` | It must be `list` or `tuple`'.format(type(ncfg[part]))
            if len(ncfg[part]) == 0:
                continue
            
            N = N_dict.get(part, None)
            if (part == 'supernode') and (N is None):
                N = len(ncfg['f'])
            netlayers = netlayers + self.__network_builder_seq(part, ncfg, N=N)
            
            pop_n = int(ncfg.get(f'pop_{part}', -1))
            for pop_n_i in range(pop_n):
                netlayers.pop()
            # Notic: other poping aproches can be considered if needed. (later)
            
        return nn.Sequential(*netlayers), [netlayers]

    def networks_builder(self, network_type: str):
        i = 0
        while True:
            neti_type = str(network_type).lower()
            neti_name = '{}{}'.format(neti_type, str(i))
            neti_config = self.kwargs.get(neti_name, None)
            if neti_config == None:
                break
            assert isinstance(neti_config, dict), '`type(neti_config)={}` | It must be a dict'.format(type(neti_config))
            neti_output = getattr(self, f'network_builder_{neti_type}')(neti_config)
            setattr(self, 'cfg_' + neti_name, neti_config)
            setattr(self, 'net_' + neti_name, neti_output[0])
            setattr(self, 'net_' + neti_name + '_detail', neti_output[1])
            
            if bool(neti_config.get('apply_weights_init', True)):
                getattr(self, 'net_' + neti_name).apply(getattr(self, neti_config.get('weights_init_fn_name', 'weights_init')))

            # print('net_' + neti_name, getattr(self, 'net_' + neti_name))
            # print('*'*60)
            i += 1

    def reparameterization(self):
        self.kwargs['forward'] = list(self.kwargs.get('forward', []))
        for i_index, i in enumerate(self.kwargs['forward']):
            if isinstance(i, dict) and len(i) == 1 and list(i.keys())[0] == 'input' and isinstance(i['input'], (list, tuple)):
                for j_index, j in enumerate(i['input']):
                    if isinstance(j, str) and j.startswith('$'):
                        self.kwargs['forward'][i_index]['input'][j_index] = self.kwargs[j[1:]]

                    if isinstance(j, (list, tuple)):
                        temp_list = []
                        for k in j:
                            if isinstance(k, str) and k.startswith('$'):
                                temp_list.append(self.kwargs[k[1:]])
                            else:
                                temp_list.append(k)
                        self.kwargs['forward'][i_index]['input'][j_index] = temp_list

                self.kwargs['forward'][i_index]['input'] = {
                    'output': i['input'][0],
                    'fnname': i['input'][1],
                    'params': i['input'][2:]
                }

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight') and hasattr(m, 'bias'):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def input_randn(self, batch_size, device, params):
        return torch.randn([batch_size] + params[0], device=device)

    def input_randint(self, batch_size, device, params):
        size = [batch_size] + params[2]
        return torch.randint(params[0], params[1], size=size, device=device) 

    def forward(self, **batch):
        batch[None] = None
        batch['None'] = None

        for current_topology in self.forward_list:
            if isinstance(current_topology, dict):
                for ctk, ctv in current_topology.items():
                    if ctk == 'concat':
                        batch[ctv['output']] = torch.cat([batch[ctv_i] for ctv_i in ctv['input']], **ctv.get('params', dict()))
                        continue
                    if ctk == 'input':
                        batch[ctv['output']] = getattr(self, f'input_{ctv["fnname"]}')(batch['batch_size'], batch['device'], ctv['params'])
                        continue
                    if ctk == 'Return':
                        if isinstance(ctv, str):
                            return batch[ctv]
                        else:
                            return [batch[r] for r in list(ctv)]
                continue
            
            current_cfg = getattr(self, 'cfg_' + current_topology)
            current_net = getattr(self, 'net_' + current_topology)
            
            if current_topology.startswith('seq'):
                batch = self.forward_seq(current_topology, batch, current_cfg, current_net, STEPWISE=self.kwargs['STEPWISE'])
                continue
            
        return batch
    
    def forward_seq(self, current_topology, batch, current_cfg, current_net, STEPWISE=False):
        saveList = []
        saveFlag = current_cfg.get('save', False)
        loadFlag = bool(current_cfg.get('load', False))
        
        if isinstance(saveFlag, (list, tuple)) and len(saveFlag) > 0:
            saveList = saveList + saveFlag
            saveFlag = True
        
        if saveFlag or loadFlag:
            STEPWISE = True

        if STEPWISE:
            x = batch[current_cfg['input']]
            for layer in current_net:
                if saveFlag and (not loadFlag):
                    x = layer(x)
                    if isinstance(layer, BB) or layer.__class__.__name__ in saveList:
                        key_save = current_topology + '_' + layer.__class__.__name__
                        batch[key_save] = batch.get(key_save, [])
                        batch[key_save].append(x)
                elif loadFlag and (not saveFlag):
                    if isinstance(layer, BB_Join):
                        layer_needed = []
                        for ln in layer.needed:
                            for ln_k, ln_v in ln.items():
                                for ln_v_i in ln_v:
                                    layer_needed.append(batch[ln_k][ln_v_i])
                        x = layer(x, *layer_needed)
                    else:
                        x = layer(x)
                elif isinstance(layer, BB_Self):
                    x = layer(x, batch)
                else:
                    x = layer(x)
            batch[current_cfg['output']] = x
        else:
            batch[current_cfg['output']] = current_net(batch[current_cfg['input']])
        
        return batch
    
    def visualize(self, visualize_fn='puml', **batch) -> dict :
        localvars = dict() 
        visited = dict()
        tag = self.kwargs['_chain_name']
        if self.kwargs['n_net'] == 1:
            tag = '.'.join(tag.split('.')[1:])
        if tag:
            tag = tag + '.'

        if visualize_fn == 'puml':
            return_fn, create_rel, create_node, get_node_id = self.__puml_init(tag)
        else:
            assert False, '`visualize_fn={}` | Does not supported | please defined `code` for this'.format(visualize_fn)
        
        for current_topology in self.forward_list:
            nid = None
            if isinstance(current_topology, dict):
                for ctk, ctv in current_topology.items():
                    if ctk != 'Return' and ctk != 'input':
                        nid = create_node(ctk, e=random_string())
                    
                    if ctk == 'input':
                        nid = create_node(ctk, e=ctv['output'], e2=ctv)
                    
                    if ctk == 'concat':
                        localvars[ctv['output']] = nid
                        for concat_inputs_i in ctv['input']:
                            lv_inp = localvars.get(concat_inputs_i, None)
                            if lv_inp is None:
                                leaf_id = create_node(concat_inputs_i, internalNodeFlag=False)
                                create_rel(l='{}{}'.format(tag, nid), r='{}{}'.format(tag, leaf_id))
                            else:
                                create_rel(r='{}{}'.format(tag, lv_inp))
                    if ctk == 'Return':
                        if isinstance(ctv, str):
                            rid = create_node(ctv, returnNodeFlag=True, update=True)
                            create_rel(l='{}{}'.format(tag, rid), returnNodeFlag=True)
                        elif isinstance(ctv, (list, tuple)):
                            u = True
                            for Rctvi in ctv:
                                rid = create_node(Rctvi, returnNodeFlag=True, update=u)
                                create_rel(l='{}{}'.format(tag, rid), returnNodeFlag=True)
                                u = False
                        else:
                            assert False, '`type(Return)={}` | It must be `str` or `list or tuple` | looking this maybe useful: `Return={}`'.format(type(ctv), ctv)
                continue
            
            current_cfg = getattr(self, 'cfg_' + current_topology)
            current_net = getattr(self, 'net_' + current_topology)
            
            if current_topology.startswith('seq'):
                seqList = getattr(self, 'net_' + current_topology + '_detail')[0]
                for idx, sl_i in enumerate(seqList):
                    nid_name = sl_i.__class__.__name__
                    endseqFlag = (idx == len(seqList)-1)
                    nid = create_node(nid_name, e=sl_i, endseqFlag=endseqFlag, outputvarname=current_cfg['output'])
                    
                    if isinstance(sl_i, BB_Join):
                        for sli_element in sl_i.needed:
                            for sl_i_ek, sl_i_ev in sli_element.items():
                                for sl_i_ev_ith in sl_i_ev:
                                    sli_topology, sli_class = sl_i_ek.split('_')
                                    sli_index = sl_i_ev_ith
                                    node_goal_counter = -1
                                    for sli_t in getattr(self, 'net_' + sli_topology):
                                        if sli_t.__class__.__name__ == sli_class:
                                            node_goal_counter += 1
                                        if node_goal_counter == sli_index:
                                            create_rel(l='{}{}'.format(tag, nid), r='{}{}'.format(tag, get_node_id(sli_t)))
                                            break
                    
                    if isinstance(sl_i, BB_Self):
                        for inp_needed in sl_i.Self_input:
                            _inpneeded = False
                            if isinstance(inp_needed, str):
                                _inpneeded = inp_needed
                            elif isinstance(inp_needed, (list, tuple)) and len(inp_needed) == 2:
                                _inpneeded = inp_needed[1]
                            elif isinstance(inp_needed, (list, tuple)) and len(inp_needed) == 1:
                                create_rel()
                                continue
                            else:
                                pass
                            
                            if _inpneeded:
                                _get_node_id_value = get_node_id(_inpneeded, flag=False)
                                if _get_node_id_value:
                                    create_rel(l='{}{}'.format(tag, nid), r='{}{}'.format(tag, _get_node_id_value))
                                else:
                                    lv_inp = localvars.get(_inpneeded, None)
                                    if lv_inp:
                                        create_rel(l='{}{}'.format(tag, nid), r='{}{}'.format(tag, lv_inp))
                                    else:
                                        print('------------->', inp_needed, _inpneeded, sl_i.Self_fn, sl_i.Self_input) # for me this never happend!
                        continue

                    if idx > 0:
                        create_rel()
                    else: # idx == 0
                        lv_inp = localvars.get(current_cfg['input'], None)
                        if lv_inp is None:
                            leaf_id = create_node(current_cfg['input'], internalNodeFlag=False, update=False)
                            create_rel(l='{}{}'.format(tag, nid), r='{}{}'.format(tag, leaf_id))
                            localvars[current_cfg['input']] = leaf_id
                        else:
                            create_rel(r='{}{}'.format(tag, lv_inp))
                localvars[current_cfg['output']] = nid
                continue
        
        return return_fn()

    
    def __puml_create_node(self, node_name, node_id=None, node_type=None, internalNodeFlag=True, returnNodeFlag=False, e=None, e2=None, endseqFlag=False, outputvarname='', update=True):
        node_param = ''
        if update:
            self.puml_temp_vars['lastnodeid'] = self.puml_temp_vars.get('currentnodeid', None)
        if (returnNodeFlag == True) or (returnNodeFlag == False and internalNodeFlag == False):
            node_type = 'enum' if node_type is None else node_type
            ID = random_string() + random_string()
        else:
            node_type = 'class' if node_type is None else node_type
            assert e is not None
            if isinstance(e, str):
                ID = e
                if e2:
                    self.puml_temp_vars['node_id__map'] = self.puml_temp_vars.get('node_id__map', dict())
                    self.puml_temp_vars['node_id__map'][e] = ID
            else:
                ID = random_string() + str(id(e))
                self.puml_temp_vars['node_id__map'] = self.puml_temp_vars.get('node_id__map', dict())
                self.puml_temp_vars['node_id__map'][str(id(e))] = ID

        if returnNodeFlag == False:
            if internalNodeFlag:
                if isinstance(e, BB_Self):
                    node_name = e.Self_fn
                node_name = pascalcase(node_name)
                
                if node_name == 'Input':
                    node_name = e
                    node_type = 'enum'
                    e2_fname = e2.get('fnname', None)
                    if e2_fname and e2_fname in ['randn', 'randint']:
                        if e2_fname == 'randn':
                            node_param = f"""
                                Noise: {e2_fname}
                                shape: {['B'] + e2['params'][0]}
                            """
                        if e2_fname == 'randint':
                            node_param = f"""
                                Noise: {e2_fname}
                                low: {e2['params'][0]}
                                high: {e2['params'][1]}
                                shape: {['B'] + e2['params'][2]}
                            """

        
        _create_node = lambda node_name, node_id, node_type, node_param='': '{} "{}" as {}{}'.format(node_type, node_name, self.puml_temp_vars['tag'], node_id) + ' \n{\n' + node_param + '\n}\n'
        
        if isinstance(e, torch.nn.Linear):
            node_param = f"""
                in_features: {e.in_features}
                out_features: {e.out_features}
            """
        if isinstance(e, torch.nn.Dropout):
            node_param = f"""
                p: {e.p}
            """
        if e.__class__.__name__ == 'FOneHot':
            node_param = f"""
                num_classes: {e.fn_params['num_classes']}
            """
        if e.__class__.__name__ == 'EinopsRearrange':
            node_param = f"""
                pattern: {e.fn_params['pattern']}
            """
        if isinstance(e, torch.nn.Softmax):
            node_param = f"""
                dim: {e.dim}
            """
        if isinstance(e, torch.nn.BatchNorm2d):
            node_param = f"""
                num_features: {e.num_features}
            """
        if isinstance(e, torch.nn.ReLU):
            node_param = f"""
                inplace: {e.inplace}
            """
        if isinstance(e, torch.nn.LeakyReLU):
            node_param = f"""
                negative_slope: {e.negative_slope}
                inplace: {e.inplace}
            """
        if isinstance(e, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            node_param = f"""
                in_channels: {e.in_channels}
                out_channels: {e.out_channels}
                kernel_size: {e.kernel_size}
                stride: {e.stride}
                padding: {e.padding}
            """
        if isinstance(e, BB_Join):
            if e.op_fn == 'cat':
                node_param = f"""
                    Operator: {e.op_fn}
                    dim: {e.kwargs.get('dim', '--')}
                """
            else:
                node_param = f"""
                    Operator: {e.op_fn}
                """
        if isinstance(e, BB_Self):
            _eparams = getattr(e, 'Self_fn_params', None)
            if _eparams is None:
                _eparams = dict()
            assert isinstance(_eparams, dict)
            _eparams = {**_eparams}
            for ep in getattr(e, 'Self_input', []):
                if isinstance(ep, str):
                    _eparams[ep] = ep
                elif isinstance(ep, (list, tuple)) and len(ep) == 1:
                    _eparams['*' + ep[0]] = 'Internal Node'
                elif isinstance(ep, (list, tuple)) and len(ep) == 2:
                    _eparams[ep[0]] = ep[1]
            node_param = ''
            for ep_k, ep_v in _eparams.items():
                node_param += f"""
                    {ep_k}: {ep_v}"""
        
        
        if endseqFlag:
            node_param = f"""
                # {outputvarname}
                {node_param}
            """
        self.puml_temp_vars['N'].append(_create_node(node_name, ID, node_type, node_param))
        
        if update:
            self.puml_temp_vars['currentnodeid'] = ID
        return ID
    
    def __puml_create_rel(self, l='', r='', returnNodeFlag=False, reverse=False):
        if returnNodeFlag:
            r = '{}{}'.format(self.puml_temp_vars['tag'], self.puml_temp_vars['lastnodeid'])
        _create_rel = lambda l, r, cur, las: (l or '{}{}'.format(self.puml_temp_vars['tag'], cur)) + self.puml_temp_vars['puml_relsign'] + (r or '{}{}'.format(self.puml_temp_vars['tag'], las))
        
        self.puml_temp_vars['R'].append(_create_rel(l, r, 
        self.puml_temp_vars['currentnodeid'],
        self.puml_temp_vars['lastnodeid']
        ))
    
    def __puml_return(self):
        return {'N': self.puml_temp_vars['N'], 'R': self.puml_temp_vars['R']}

    def __puml_get_node_id(self, obj, flag=True):
        if flag:
            return self.puml_temp_vars['node_id__map'][str(id(obj))]
        else:
            return self.puml_temp_vars['node_id__map'].get(obj, None)
    
    def __puml_init(self, tag):
        self.puml_temp_vars = dict()
        self.puml_temp_vars['N'] = []
        self.puml_temp_vars['R'] = []
        self.puml_temp_vars['puml_relsign'] = ' ' + str(self.kwargs.get('puml_relsign', '<|--')) + ' '
        self.puml_temp_vars['tag'] = tag
        return self.__puml_return, self.__puml_create_rel, self.__puml_create_node, self.__puml_get_node_id

