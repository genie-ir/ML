import importlib
import os, sys, re, cowsay
from libs.basicHR import EHR
from libs.basicIO import pathBIO, readBIO

def __imp(adr_path, package, embedParams):
    embedParams = dict() if embedParams is None else embedParams
    try:
        m = importlib.import_module(adr_path, package=package)
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            cowsay.cow('ModuleNotFoundError:\nplease define: `{}`'.format(adr_path))
            sys.exit()
        else:
            assert False, e
    for k in embedParams:
        setattr(m, k, embedParams[k])
    return m

def Import(fpath, reload=False, partialFlag=True, package=None, embedParams=None):
    if partialFlag: # e.g. fpath="articles.taming_transformers.taming.models.vqgan.VQModel"
        _module, _cls = fpath.rsplit('.', 1)
        if reload:
            module_imp = importlib.import_module(_module)
            importlib.reload(module_imp)
        return getattr(__imp(_module, package=package, embedParams=embedParams), _cls)
    else:
        return __imp(fpath, package=package, embedParams=embedParams)

def instantiate_from_config(config, kwargs=None):
    kwargs = dict() if kwargs is None else kwargs
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    if config['target'].endswith('.yaml') or config['target'].endswith('.json'):
        ext = config['target'][-5:]
        config['target'] = config['target'].replace('.', os.sep)
        fpath = pathBIO('//' + os.path.split(config['target'])[0]) + ext
        if os.path.exists(fpath):
            return readBIO(fpath, **kwargs)
        else:
            return None
    elif config['target'].endswith('.dont'):
        config['target'] = config['target'][:-5]
        return Import(config['target'])
    elif config['target'].endswith('.ignore'):
        return None
    else:
        return Import(config['target'])(**{**kwargs, **config.get('params', dict())})
        # return Import(config['target'])(**config.get('params', dict()), **kwargs)


# def Import(path, fname=None):
#     path = pathBIO(path)
#     fname = path.split('/')[-1].replace('.py', '')
#     return fname
#     try:
#         spec = importlib.util.spec_from_file_location(fname, path)
#         f = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(f)
#         return f
#     except Exception as e:
#         EHR(e)