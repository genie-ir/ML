from os import environ
from os.path import expanduser

cacheDir = lambda: environ.get('XDG_CACHE_HOME', expanduser('~/.cache'))