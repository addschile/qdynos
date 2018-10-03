def print_method(line):
    nchar = len(line)
    print("#"*40)
    print("#"+" "*int((40-2-nchar)/2)+line+" "*int((40-2-nchar)/2)+" #")
    print("#"*40)
    print("")

def print_stage(line):
    print("==> %s <=="%(line))
