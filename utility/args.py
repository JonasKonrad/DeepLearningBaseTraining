import os
import argparse as ar
from typing import Any
import tomllib
from difflib import get_close_matches

DEFAULTS_FILE_PATH = "utility/defaults.toml"

class MetaClass(type):
    def __getattr__(cls, key: str) -> Any:
        return cls.get(key)

class Args(metaclass=MetaClass):
    """
    Argument parsing used from default argpase module. Changes:
        - defaults are not allowed in source code
        - a default value must be provided for each argument in the default file (see DEFAULTS_FILE_PATH)
        - bool args are handeled as follows:
            -> --arg --> True
            -> --arg True --> True
            -> --arg False --> False
            -> (not given) --> default from ini file
        - list arguments (nargs="*") can be added/removed by parsing '--NAME+ a,b,c' or '--NAME- d,e'
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

        with open(DEFAULTS_FILE_PATH, "rb") as f:
            defaults = tomllib.load(f)

        ## set defaults
        # check if all defaults are existing
        for action in cls.parser._actions:
            if not isinstance(action, ar._HelpAction):
                if action.dest not in defaults:
                    raise RuntimeError(f"No default argument given for {action.dest}. Please add to '{DEFAULTS_FILE_PATH}'")
        # check if all file elements where defined
        for key in defaults.keys():
            if key not in [action.dest for action in cls.parser._actions if not isinstance(action, ar._HelpAction)]:
                raise RuntimeError(f"Parameter {key} were given in default file but not defined before.")

        cls.parser.set_defaults(**defaults)

        #only parse input file name
        init_file_name_parser = ar.ArgumentParser(description="Blank Project",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", metavar="FILE")
        ifile_args, remaining_argv = init_file_name_parser.parse_known_args()
        ifile_path = ifile_args.ifile

        if ifile_path is not None:
            with open(ifile_path, "rb") as f:
                ifile_arguments = tomllib.load(f)

            # check if all file elements where defined
            for arg in ifile_arguments.keys():
                if arg not in [action.dest for action in cls.parser._actions if not isinstance(action, ar._HelpAction)]:
                    raise RuntimeError(f"Input file argument '{arg}' not found. Did you mean '{get_close_matches(arg, list(cls.parser._defaults.keys()), n = 1, cutoff = 0)[0]}'?")

            cls.parser.set_defaults(**ifile_arguments)

        # parse command line arguments
        cls.data, remaining_argv = cls.parser.parse_known_args(remaining_argv)

        # if singular values are given for list type arguments in default or ini file, these might not be parsed correctly. thus need to transform those to lists of single elements. 
        for action in cls.parser._actions:
            if action.nargs == "*" and not isinstance(getattr(cls.data, action.dest), list):
                setattr(cls.data, action.dest, [getattr(cls.data, action.dest), ])

        # parse add/remove arguments for lists
        remaining_argv = cls.parse_special_list_args(remaining_argv)

        # check for unrecogniced command line arguments
        for arg in remaining_argv:
            raise RuntimeError(f"Command line argument '{arg}' not found. Did you mean '{get_close_matches(arg, list(cls.parser._defaults.keys()), n = 1, cutoff = 0)[0]}'?")

        # expand possibly occurring "~" in path arguments. @TODO make this not manually for each path argument?! 
        cls.data.dataDir = os.path.expanduser(cls.data.dataDir)
        cls.data.logSubDir = os.path.expanduser(cls.data.logSubDir)
        cls.data.logDir = os.path.expanduser(cls.data.logDir)

    @classmethod
    def parse_special_list_args(cls, argsv: list) -> None:
        """ function allowing to add or remove entries from list arguments by parsing '--NAME+ a,b,c' or '--NAME- d,e'"""
        i = 0
        while i < len(argsv):
            if argsv[i].endswith('+'):
                add = True
            elif argsv[i].endswith('-'):
                add = False
            else:
                i += 1
                continue

            name = argsv[i][2:-1]
            values = argsv[i+1].split(",")

            # find action
            for action in cls.parser._actions:
                if action.dest == name:
                    break
            else:
                raise RuntimeError(f"Action not found for arg '{argsv[i]}'")

            if action.nargs != "*":
                raise RuntimeError(f"add/sub not allowed for non list option '{argsv[i]}'")
            
            for val in values:
                if add:
                    getattr(cls.data, name).append(action.type(val))
                else: #remove
                    getattr(cls.data, name).remove(action.type(val))
            
            del argsv[i+1]
            del argsv[i]
        return argsv

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
