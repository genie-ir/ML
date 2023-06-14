import os, libtmux
from time import sleep

import hashlib
from random import choice
from string import ascii_uppercase

md5 = lambda s: hashlib.md5(s).hexdigest()
sha1 = lambda s: hashlib.sha1(s.encode('utf-8')).hexdigest()
sha256 = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()

random_string = lambda L=32: ''.join(choice(ascii_uppercase) for i in range(L))


class TMUXBase():
    """
        Example:
            tmux_obj = TMUXBase()
            pane = tmux_obj.get_pane(session_obj='teh', window_name='comp', pane_id='%9')
            tmux_obj.input(pane, "cowsay 'hello'") # ruu on the terminal
            tmux_obj.input(pane, "cowsay 'hello'", enter=False) # just type to terminal and not run it.
            out = tmux_obj.output(pane)
    """
    def __init__(self):
        os.system('tmux start-server')
        self.server = libtmux.Server()
        self.__sessions = dict() # self_created_sessions not all sessions. # just stores created sessions by this instance of class.

    def get_server(self):
        return self.server
    
    def get_all_sessions(self):
        return self.server.sessions
    
    def get_all_sessions_names(self):
        all_sessions = self.get_all_sessions()
        return [si.name for si in all_sessions]

    def get_self_created_sessions(self, key=None):
        if key:
            return self.__sessions.get(key, None)
        else:
            return self.__sessions
    
    def filter(self, **kwargs):
        return self.server.sessions.filter(**kwargs)
    
    def get(self, **kwargs):
        return self.server.sessions.get(**kwargs)
    
    def get_session_with_name(self, session_name, index=0, saveFlag=True):
        if index == -1:
            s = self.server.sessions.filter(session_name=session_name)
        else:
            s = self.server.sessions.filter(session_name=session_name)[index]
        if saveFlag:
            self.__sessions[session_name] = s
        return s

    def create_session(self, name=None):
        name = name if name else sha1(random_string())
        all_sessions_names = self.get_all_sessions_names()
        assert not (name in all_sessions_names), 'name: `{}` is already exist among sessions. plese choose a unique name for session.'.format(name)
        os.system('tmux new-session -d -s "{}"'.format(name))
        return self.get_session_with_name(name)
    
    def __private_get_session_obj(self, session_obj=None):
        sessions_keys = list(self.__sessions.keys())
        if isinstance(session_obj, str):
            if session_obj in sessions_keys:
                return self.__sessions[session_obj]
            elif session_obj in self.get_all_sessions_names():
                _all_sessions = self.get_all_sessions()
                for si in _all_sessions:
                    if si.name == session_obj:
                        return si
            else:
                assert False, 'session=`{}` does not exist.'.format(session_obj)
        if session_obj:
            return session_obj
        elif session_obj is None and len(sessions_keys) == 1:
            return self.__sessions[sessions_keys[0]]
        else:
            assert False, 'session_obj `{}` is not valid.'.format(session_obj)

    def rename_session(self, new_name: str, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.rename_session(str(new_name))
    
    def get_attached_window(self, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.attached_window
    
    def create_window(self, window_name: str, session_obj=None, attach=False):
        window_name = str(window_name)
        assert window_name.lower() != '[tmux]', '`[tmux]` is not valid for window_name'
        session_obj = self.__private_get_session_obj(session_obj)
        session_obj_window_names = [w.name for w in session_obj.windows]
        if window_name in session_obj_window_names:
            assert False, 'window name: `{}` already exist and its better choose unique name!'.format(window_name)
        else:
            return session_obj.new_window(attach=attach, window_name=window_name)
    
    def kill_window(self, window_name: str, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        return session_obj.kill_window(str(window_name))

    def get_window(self, name=None, wid=None, session_obj=None):
        session_obj = self.__private_get_session_obj(session_obj)
        if wid == None and name == None:
            if len(session_obj.windows) == 1:
                return session_obj.windows[0]
            else:
                return session_obj.windows
        else:
            if name is not None:
                name = str(name)
                while True:
                    copy_session_obj_windows = session_obj.windows
                    if not ('[tmux]' == str(session_obj.attached_window.name).lower()):
                        break
                    try:
                        session_obj.attached_pane.display_message('This session belongs to the process with PID={}, please don\'t make any changes and stop watching it, any tiny action of you may affect on the flow of the process.'.format(os.getppid()))
                    except Exception as e:
                        pass
                    sleep(1)
                sum_var = sum([1 for wi in copy_session_obj_windows if wi.name == name])
                if sum_var == 1:
                    for w in copy_session_obj_windows:
                        if w.name == name:
                            return w
                elif sum_var == 0:
                    return None # not found
                else:
                    assert False, '(multiple window found) | name: `{}` is not unique in session_name: `{}` and you must search with `window_id` attribute. | windows_list: `{}`'.format(name, session_obj.name, copy_session_obj_windows)
            
            if wid is not None:
                for w in session_obj.windows:
                    if w.id == wid:
                        return w
                return None
            
            return None
    
    def input(self, pane_real_obj, *pargs, **kwargs):
        pane_real_obj.clear()
        while True:
            try:
                if '@' in pane_real_obj.capture_pane()[-1]:
                    break
            except Exception as e:
                pass
            sleep(1)
        pane_real_obj.send_keys(*pargs, **kwargs)

    def output(self, pane_real_obj, printFlag=False, ignore_first_and_last_lines=True):
        out = '\n'.join(pane_real_obj.cmd('capture-pane', '-p').stdout)
        if ignore_first_and_last_lines:
            out = '\n'.join(out.split('\n')[1 : -1])
        if printFlag:
            print(out)
        return out

    def get_pane(self, window_name=None, window_id=None, window_real_obj=None, pane_id=None, session_obj=None):
        """easy way is just determine window_name"""
        if window_real_obj is not None:
            w = window_real_obj
        else:
            session_obj = self.__private_get_session_obj(session_obj)
            w = self.get_window(name=window_name, wid=window_id, session_obj=session_obj)
        if w is not None:
            if isinstance(w, (list, tuple)):
                assert False, 'w is a list or tuple: `{}` you must exactly determine which w is needed with `window_name` or `window_id` attributes.'.format(w)
            if len(w.panes) == 1:
                return w.panes[0]
            else:
                if pane_id is not None:
                    for pi in w.panes:
                        if pi.id == pane_id:
                            return pi
                    assert False, 'pane_id: `{}` is not exist in window_name: `{}` of session_name: `{}`.'.format(pane_id, w.name, w.session_name)
                else:
                    assert False, '(multiple pane) | there are `{}` panes are exist inside window_name: `{}` of session_name: `{}`. | you can use pain_id to determine pane of pain_list: `{}`.'.format(len(w.panes), w.name, w.session_name, w.panes)
        else:
            assert False, 'window_name: `{}` is not exist in session_name: `{}`'.format(window_name, session_obj.name)


# t = TMUXBase()
# s = t.create_session()
# pr = t.create_window(window_name="reg").attached_pane
# ps = t.create_window(window_name="sup").attached_pane
# t.input(pr, 'pipenv shell')
# t.input(pr, 'echo $PS1')
# t.input(ps, 'sudo su')
# ps.send_keys('narnea')
# t.input(ps, 'pipenv shell')
# print(pr.capture_pane()[-2])



# fname = '/tmp/{}.txt'.format(sha1(random_string()))
# p.send_keys('echo $PS1 > {}'.format(fname))
# t.input(p, 'exit')

# t.input(p, 'pipenv shell')
# p.send_keys('PS1=$(cat {})'.format(fname))
