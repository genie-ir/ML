import numpy as np
import pandas as pd
from dotted_dict import DottedDict
# from attrdict import AttrDict # Not work in python 3.10
from argparse import Namespace
from functools import partial, wraps
from libs.basicEX import KeyNotFoundError
from typing import Any, Callable, Optional, Union

# def ns_add(ns: Namespace, *posargs: Union[Namespace, dict]):
#     pass

def dotdict(d: dict, flag=None):
    flag = bool(True if flag is None else flag)
    if isinstance(d, dict) and flag:
        return DottedDict(d)
    return d

def dict2ns(d: dict):
    if isinstance(d, dict):
        return Namespace(**d)



# Example: d does not contain dict in list!! (like below) d['d'][3] is not support By this function!!
# d = {'a': 1,
#      'c': {'a': '#a_val', 'b': {'x': '#x_value', 'y' : '#y', 'z': {'aa': 12, 'cv': 13}}},
#      'd': [1, '#d_i1', 3, {'b': {'x': '#x_value', 'y' : '#y'}}]}
# 
# sep = '_'
# d2 = make_flatten_dict(d, sep=sep)
# print(d2)
# make_nested_dict(d2, sep=sep)
# Outputs:
# {'a': 1, 'd': [1, '#d_i1', 3, {'b': {'x': '#x_value', 'y': '#y'}}], 'c_a': '#a_val', 'c_b_x': '#x_value', 'c_b_y': '#y', 'c_b_z_aa': 12, 'c_b_z_cv': 13}
# 
# {'a': 1,
#  'd': [1, '#d_i1', 3, {'b': {'x': '#x_value', 'y': '#y'}}],
#  'c': {'a': '#a_val',
#   'b': {'x': '#x_value', 'y': '#y', 'z': {'aa': 12, 'cv': 13}}}}

def make_flatten_dict(d, sep='.'):
    return pd.json_normalize(d, sep=sep).to_dict(orient='records')[0]

def make_nested_dict(flatted_dict, sep='.'):
    result = {}
    for k, v in flatted_dict.items():
        tmp = result
        *keys, last = k.split(sep)
        for key in keys:
            tmp = tmp.setdefault(key, {})
        tmp[last] = v
    return result

def retrieve(
    list_or_dict, key, splitval='/', default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Example:
        t = retrieve({'size': 256, 'ImageNetTrain': {'random_crop': 'ok'}, 'hoo': ['ali', {'random_crop': 'sasan'}]}, "hoo/1/random_crop",
                                    default=True)

        print('t', t)

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            'Trying to get past callable node with expand=False.'
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


