import torch
from os import getenv, makedirs
from libs.basicIO import compressor
from utils.plots.plot1d import Plot1D
from libs.basicTime import getTimeHR_V0
from os.path import join, split as SplitPath

def fwd_plot(batch: torch.Tensor, plot_params, sigkey, useCompressorFlag=True):
    FFT_PASSIVE_FN = (getenv('PLOT_DEFAULT_FFT_PASSIVE_FN') or ':'.join(['real', 'imag', '..abs', '..angle'])).split(':')
        
    if isinstance(plot_params, str):
        plot_params = {'type': plot_params}
    assert isinstance(plot_params, dict), '`type(plot_params)={}` | It must be `str` or `dict`'.format(type(plot_params))

    label = lambda l: str(l).replace('.', '')

    plot_type, plot_style = str(plot_params.get('type', getenv('PLOT_DEFAULT_TYPE') or 'plot1D:neon')).split(':')
    plot_path = plot_params.get('path', getenv('PLOT_DEFAULT_PATH') or join(getenv('GENIE_ML_REPORT'), 'plot', getTimeHR_V0()))
    makedirs(plot_path, exist_ok=True)
    
    plot_x, plot_y = plot_params.get('xlabel', 'x'), plot_params.get('ylabel', 'y')
    plot_smoothing = plot_params.get('smoothing', True)
    plot_params['params'] = list(plot_params.get('params', [
        None, # this convert to empty dict automaticaly
        {'ffty': True, 'ffty_div2N': True, 'passive_fn': '*'} # here `*` it means all available `fft` passive_fns
    ]))
    for ppp_ith in range(len(plot_params['params'])):
        if isinstance(plot_params['params'][ppp_ith], dict) and (plot_params['params'][ppp_ith].get('smoothing', None) is None):
            plot_params['params'][ppp_ith]['smoothing'] = plot_smoothing

    if plot_type == 'plot1D':
        Bskey = batch[sigkey].flatten(start_dim=1).detach().cpu().numpy()
        for idx_s, s in enumerate(Bskey):
            sx = str(batch['x'][idx_s])
            sy = batch['y'][idx_s].detach().cpu().item()
            sp = join(plot_path, str(sy), '@', '{}.png'.format(SplitPath(sx)[1].split('.')[0]))

            for _plotparams_i in plot_params['params']:
                lbl = None
                plotparams_i = dict()
                if isinstance(_plotparams_i, dict):
                    plotparams_i = _plotparams_i
                plot_passive_fn = plotparams_i.get('passive_fn', None)
                
                ###################################### signal
                if len(plotparams_i) == 0 or plotparams_i.get('space', None) == 'time':
                    lbl = label(plotparams_i.get('label', 'signal'))
                    plot1d = Plot1D(xlabel=plot_x, ylabel=plot_y, mplstyle=plot_style)
                    plot1d.plot(y=s, label=lbl, **plotparams_i)
                    plot1d.savefig(sp.replace('@', lbl))
                    continue
                
                ###################################### fft
                if plotparams_i.get('ffty', False):
                    if plot_passive_fn == '*':
                        for _fft_passive_fn in FFT_PASSIVE_FN:
                            plotparams_j = {**plotparams_i}
                            plotparams_j['passive_fn'] = _fft_passive_fn
                            lbl = label(plotparams_i.get('label', _fft_passive_fn))
                            plot1d = Plot1D(xlabel=plot_x, ylabel=plot_y, mplstyle=plot_style)
                            plot1d.plot(y=s, label=lbl, **plotparams_j)
                            plot1d.savefig(sp.replace('@', lbl))
                    else:
                        lbl = label(plotparams_i.get('label', plot_passive_fn))
                        plot1d = Plot1D(xlabel=plot_x, ylabel=plot_y, mplstyle=plot_style)
                        plot1d.plot(y=s, label=lbl, **plotparams_i)
                        plot1d.savefig(sp.replace('@', lbl))
                    continue
    

    if useCompressorFlag:
        compressor(src_dir=plot_path, dst_file=join(plot_path + '.zip'))
                