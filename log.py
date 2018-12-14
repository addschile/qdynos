import sys

def print_basic(line):
    sys.stdout.write(line)
    sys.stdout.flush()

def print_method(line):
    nchar = len(line)
    sys.stdout.write("#"*40+"\n")
    sys.stdout.write("#"+" "*int((40-2-nchar)/2)+line+" "*int((40-2-nchar)/2)+" #"+"\n")
    sys.stdout.write("#"*40+"\n")
    sys.stdout.flush()

def print_stage(line):
    sys.stdout.write("==> %s <==\n"%(line))
    sys.stdout.flush()

def print_progress(percent,tau):
    sys.stdout.write("%.0f Percent done"%(percent)+"."*10+"%.8f\n"%(tau))
    sys.stdout.flush()

def print_time(tau):
    sys.stdout.write("Elapsed time: %.8f\n"%(tau))
    sys.stdout.flush()

# TODO add classes to handle logging
#class Log:
#    import logging
#    def __init__(self,):

# TODO add restart file system
#class Restart:
#    import h5py
#
#    def __init__(self, file_name):
#
#    def write_matrix():
#
#    def write_vector():
#
#    def read_matrix():
#
#    def read_vector():
