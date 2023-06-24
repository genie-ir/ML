import numpy as np

def psvfn(t: np.ndarray, passive_fns):
    """apply passive_fns to np.array: `t` step by step"""
    
    if passive_fns is None:
        print('p'*30)
        return t
    
    print('????????????????????')
    
    if not isinstance(passive_fns, (list, tuple)):
        passive_fns = [passive_fns]
    
    for _psvfn in passive_fns:
        psvfn  = None
        psvval = dict()
        
        if isinstance(_psvfn, dict):
            assert len(_psvfn) == 1, '`len(_psvfn)={}` It must be `1`'.format(len(_psvfn))
            _psvfn_key = str(list(_psvfn.keys())[0])
            if not _psvfn_key.startswith('.'):
                psvfn = '..' + _psvfn_key
            else:
                psvfn = _psvfn_key
            psvval = _psvfn[_psvfn_key]
        elif isinstance(_psvfn, str):
            psvfn = _psvfn
        else:
            assert False, '`type(_psvfn)={}` | It must be `str` or `dict`'.format(type(_psvfn))
        
        if isinstance(psvfn, str):
            if psvfn.startswith('..'):
                t = eval(f'np.{psvfn[2:]}(t, **psvval)')
            elif psvfn.startswith('.'):
                t = eval(f't.{psvfn[1:]}(**psvval)')
            else:
                t = eval(f't.{psvfn}')
    
    return t