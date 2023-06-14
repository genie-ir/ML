from functools import wraps
from typing import Any, Callable, Optional


def log(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if True:
            print(f'function [{fn.__name__}] called')
            return fn(*args, **kwargs)
        return None

    return wrapped_fn

@log
def mmd(name, age):
    return f'name: {name} and age: {age}'


print(mmd('harry', 23))
print(mmd.__name__)