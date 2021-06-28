"""
python module that will do pretty printing with colors to stdout. 
"""
import sys
import numpy as np

np.set_printoptions(linewidth=250)

class colors:
    """
    Class that contains colors.
    """
    ENDC      = "\033[0m"
    WHITE     = "\033[0m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"
    ORANGE    = "\033[31m"
    BLUE      = "\033[34m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    PURPLE    = "\033[94m"
    PINK      = "\033[95m"
    CYAN      = "\033[96m"
    GRAY      = "\033[37m"
    BLACK     = "\033[30m"

def pprint(color, text, newline=True):
    """
    Function that prints text to the system in different colors.

    Parameters
    ----------
    color: string
        given in form colors.WHATEVERCOLOR - pick text color
    text: string
        whatever you want it to say
    newline: bool
        whether to put a linebreak after the text

    Example
    -------
    from pretty_printing import pprint
    pprint( orange, "Hello world!")
    """
    sys.stdout.write( color + text + colors.ENDC )
    if newline: sys.stdout.write( "\n" )
    return 


