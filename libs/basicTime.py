import time
import datetime

def getTimeHR(timestamp=None, split: str =' ', dateFormat: str ='%Y-%m-%d', timeFormat: str ='%H:%M:%S', now: bool =False):
    if now:
        return datetime.datetime.now().strftime(f'{dateFormat}{split}{timeFormat}')
    else:    
        timestamp = timestamp if timestamp else time.time()
        return datetime.datetime.fromtimestamp(timestamp).strftime(f'{dateFormat}{split}{timeFormat}')

def getTimeHR_V0():
    return getTimeHR(now=True, split='T', dateFormat='%Y-%m-%d', timeFormat='%H-%M-%S')