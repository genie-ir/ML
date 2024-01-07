# PEP 8 recommends using lowercase letters for variable and function names, 
# with words separated by underscores. For class names, use the CamelCase convention, 
# starting with an uppercase letter. It is also recommended to avoid using 
# single-character names unless they are for temporary or looping variables.

from types import MethodType, FunctionType

class PYBASE:
    """
        # NOTE: if you wanna define params for this function all of those shoude be named as `PYB_varname`.
        becareful in naming variable and function becuse this class can be inherit in multi class inheritence 
        starategy as left as possible and will be overwite all other right classes.
    """
    def __init__(self, *pargs, **kwargs):
        try:
            super().__init__(*pargs, **kwargs) # NOTE: this line shoulde be defined. in multi class inheritence, inherit this class as `left` as possible.
        except Exception as e:
            if str(e) == 'object.__init__() takes exactly one argument (the instance to initialize)':
                super().__init__() # NOTE: there is no multi inheritence and we just ready for to go `object` class with no parammeters.
            else: # NOTE: there is multi inheritence and exception was acourd in that class and we should raise this exception becuse its not my problem!
                raise Exception(e)

        # NOTE: from heare and below all variables and functions you defined will be overwrite all variables and functions defined in other right classes in multi class inheritence strategy.
        self.pargs = pargs
        self.kwargs = kwargs
        self.__start()

    def __start(self):
        pass
    
    # def def_ifn(self, name: str, fn, **kwargs):
    #     """
    #         define instance function in specefic or all object(s).
    #         fn is must be define outside of any classes and it must be has `self` param as first in its parammeters.
    #     """
    #     # setattr(self.__class__, name, fn) # define in all instances
    #     Self = kwargs.get('Self', self) # OPTIONAL
    #     setattr(Self, name, MethodType(fn, Self)) # define in one instance