### global constants ###
hbar = None

### energy units ###
cm2au = 4.556e-6 # H/cm^-1
ev2au = 0.0367493 # H/eV

### mass units ###
me2au = 0.00054858 # amu/m_e

### distance units ###
ang2bohr = 1.88973 # bohr/angN

### time units ###
fs2au = 41.3413745758 # au/fs

def working_units():
    from numpy import isclose 
    if hbar==None:
        print("No working units assigned\n")
    elif isclose(hbar,5308.8):
        print("ENERGY UNITS: cm^-1\nTIME UNITS: fs\n")
    elif isclose(hbar,0.6582):
        print("ENERGY UNITS: eV\nTIME UNITS: fs\n")
    else:
        print("Unrecognized units system\n")
