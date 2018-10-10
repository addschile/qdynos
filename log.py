import std

def print_method(line):
    nchar = len(line)
    sys.stdout.write("#"*40+"\n")
    sys.stdout.write("#"+" "*int((40-2-nchar)/2)+line+" "*int((40-2-nchar)/2)+" #"+"\n")
    sys.stdout.write("#"*40+"\n")
    sys.stdout.flush()

def print_stage(line):
    sys.stdout.write("==> %s <=="%(line))
    sys.stdout.flush()

# TODO add classes to handle logging
#class Log:
#    import logging
#    def __init__(self,):

# TODO add restart file system
#class Restart:
#    import h5py
#
#    def __init__(self,):
