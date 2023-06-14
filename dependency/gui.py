import pretty_errors
import sys
import pathlib
from os.path import join

def rootPath():
    return pathlib.Path(__file__).parents[1]

def pathBIO(fpath: str, **kwargs):
    if fpath.startswith('//'):
        fpath = join(rootPath(), fpath[2:])
    return fpath

# import tkinter as tk
# from tkinterweb import HtmlFrame
# root = tk.Tk() #create the tkinter window
# frame = HtmlFrame(root) #create the HTML browser
# frame.load_url(
#     # '/home/alihejrati/Documents/code/ML/ui/graph/index.html'
#     'file://' + pathBIO(join('//', 'ui', 'graph', 'index.html'))
# ) #load a website
# frame.pack(fill="both", expand=True) #attach the HtmlFrame widget to the parent window
# root.mainloop()


import webview
webview.create_window('Hello world', 'file://' + pathBIO(join('//', 'ui', 'graph', 'index.html')))
webview.start()