import pudb
import signal
from loguru import logger
from os import getenv, getpid, kill

def EHR(e):
    logger.error(e)
    if getenv('GENIE_ML_DEBUG_MODE') == 'True':
        pudb.post_mortem()
        raise e

class SHR:
    __sig = dict()
    def __init__(self, **kwargs):
        """
        Example:
            def SIGUSR1(sn, frame): # function name not important.
                print('ok', sn, frame)
                signal.pause()

            SHR(SIGUSR1=SIGUSR1, SIGUSR2=None)
        """
        for key in kwargs:
            if key.startswith('SIG'):
                handler_fn = kwargs[key] if kwargs[key] else signal.SIG_IGN
                SHR.__sig[key] = signal.signal(getattr(signal, key), handler_fn)
        
        if kwargs.get('showPidFlag', True):
            logger.info(f'My PID is: {getpid()}')
        if kwargs.get('pauseFlag', False):
            signal.pause()
    
    @classmethod
    def send(cls, sig=None, pid=None):
        pid = pid if pid else getpid()
        sig = sig if sig else signal.SIGUSR1
        if isinstance(sig, str):
            sig = getattr(signal, sig)
        return kill(pid, sig)