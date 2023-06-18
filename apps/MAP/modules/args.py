from utils.args import ParserBasic, str2bool
from argparse import ArgumentParser, ArgumentTypeError

class Parser(ParserBasic):
    @classmethod
    def get_parser(cls, **kwargs):
        parser = super().get_parser(**kwargs)
        # add your arguments to parser...
        return parser
    
    @classmethod
    def ctl_parser(cls, opt, unknown, **kwargs):
        return super().ctl_parser(opt, unknown, **kwargs)