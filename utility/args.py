import configparser
import argparse as ar
from typing import Any
import sys
from difflib import get_close_matches

DEFAULTS_FILE_PATH = "utility/defaults.ini"

class MetaClass(type):
    def __getattr__(cls, key: str) -> Any:
        return cls.get(key)

class Args(metaclass=MetaClass):
    """
    Argument parsing used from default argpase module. Changes:
        - defaults are not allowed in source code
        - a default value must be provided for each argument in the input file
        - bool args are handeled as follows:
            -> --arg --> True
            -> --arg True --> True
            -> --arg False --> False
            -> (not given) --> default from ini file
    """
    data: ar.Namespace = None
    parser = ar.ArgumentParser()
    
    @classmethod
    def get(cls, key: str) -> Any:
        if cls.data is None:
            raise RuntimeError(f"Arguments were not parsed yet.")
        return getattr(cls.data, key)
    
    @classmethod
    def add_argument(cls, *args, **kwargs) -> None:
        if "default" in kwargs:
            raise RuntimeError("Defining defaults in source code is prohibited. Use input file instead.")
        if "type" in kwargs:
            if kwargs["type"] is bool:
                kwargs["nargs"] = "?"
                kwargs["const"] = True
        cls.parser.add_argument(*args, **kwargs)
    
    @classmethod
    def parse_args(cls) -> None:
        # replace constructor class for bool type arguments with custom function
        for action in cls.parser._actions:
            if action.type is bool:
                action.type = make_bool

        config = configparser.ConfigParser()
        config.optionxform = str # make config parse case sensitive
        out = config.read([DEFAULTS_FILE_PATH])
        if not out:
            raise RuntimeError(f"Default file '{DEFAULTS_FILE_PATH}' not found.")

        defaults = {}
        for sec in config.sections():
            defaults.update(**dict(config[sec]))
        cls.set_defaults(defaults, mustExist = True)

        #only parse input file name
        init_file_name_parser = ar.ArgumentParser(description="Blank Project",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", metavar="FILE")
        ifile_args, remaining_argv = init_file_name_parser.parse_known_args()
        ifile_path = ifile_args.ifile
        
        if ifile_path is not None:
            out = config.read([ifile_path])
            if not out:
                raise RuntimeError(f"Ifile '{ifile_path}' not found.")
            
            ifile_arguments = {}
            for sec in config.sections():
                ifile_arguments.update(**dict(config[sec]))
            
            cls.set_defaults(ifile_arguments)

        cls.data, still_remaining_argv = cls.parser.parse_known_args(remaining_argv)

        # review not-parsed arguments 
        if still_remaining_argv:
            possible_arguments = list(cls.parser._defaults.keys())
            missmatch_string = ""
            for arg in still_remaining_argv:
                candidate, = get_close_matches(arg, possible_arguments, n = 1, cutoff = 0)
                missmatch_string += f"\n{arg[2:]} --> {candidate}"
            raise RuntimeError(f"Command line arguments '{', '.join(still_remaining_argv)}' not found. Possible missmatches: {missmatch_string}")
        

    @classmethod
    def set_defaults(cls, defaults, mustExist = False) -> None:
        # set defaults
        default_keys = list(defaults.keys())
        for action in cls.parser._actions:
            if not isinstance(action, ar._HelpAction):
                if mustExist and action.dest not in defaults:
                    raise RuntimeError(f"No default argument given for {action.dest}. Please add to '{DEFAULTS_FILE_PATH}'")
                
                if action.nargs == "*":
                    # split lists seperated by whitespace in defaults file
                    defaults[action.dest] = defaults[action.dest].split(" ") if defaults[action.dest] else []

                default_keys.remove(action.dest)

        if default_keys:
            raise RuntimeError(f"Parameter(s) {default_keys} were given in input file but not defined before.")
        
        cls.parser.set_defaults(**defaults)
        

    @classmethod
    def parse_args_contin(cls, defaults: dict) -> None:
        defaults["contin"] = True #make sure contin argument is not overwritten by defaults from param.json file
        cls.parser.set_defaults(**defaults) #read defaults from param.json file
        
        # parse ifile argument here only to not throw an error by real parser
        init_file_name_parser = ar.ArgumentParser(description="",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", default='input.ini',metavar="FILE")
        _, remaining_argv = init_file_name_parser.parse_known_args()
        
        cls.data = cls.parser.parse_args(remaining_argv)



def make_bool(arg: Any) -> bool:
    if isinstance(arg, str):
        if arg.lower() in ['true', '1', 't', 'y', 'yes']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'n', 'no']:
            return False
        else:
            raise RuntimeError(f"Could not convert string to bool: {arg}")
    else:
        return bool(arg)
