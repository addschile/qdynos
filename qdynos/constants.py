### global constants ###
hbar = None

### dictionary for hbar ###
hbars = {'ev': 0.658229,
         'cm': 5308.8,
         'au': 1.0
        }

### dictionary for kb ###
kbs = {'ev': 8.597924640741457e-05,
       'cm': 0.69352,
       'au': 3.15967712e-06
      }

### energy units ###
cm2au = 4.556e-6 # H/cm^-1
ev2au = 0.0367493 # H/eV
au2cm = 1./cm2au
au2ev = 1./ev2au

### mass units ###
me2au = 0.00054858 # amu/m_e

### distance units ###
ang2bohr = 1.88973 # bohr/ang

### time units ###
fs2au = 41.3413745758 # au/fs
ps2au = fs2au/1000. # au/ps
fs2ps = 1./1000. # ps/fs

def working_units():
  from numpy import isclose 
  if hbar==None:
    print("No working units assigned\n")
  elif isclose(hbar,5308.8):
    print("ENERGY UNITS: cm^-1\nTIME UNITS: fs\n")
  elif isclose(hbar,0.658229):
    print("ENERGY UNITS: eV\nTIME UNITS: fs\n")
  elif hbar==1.0:
    print("ENERGY UNITS: au\nTIME UNITS: au\n")
  else:
    print("Unrecognized units system\n")

# TODO
#def convert_units():
    
def get_hbar(*units):
  if len(units)==0:
    return hbars['au']
  else:
    try:
      assert(units[0].lower() in ['ev','cm','au'])
    except:
      raise ValueError('Not a valid units system')
    return hbars[units[0].lower()]

# TODO this isn't working
def set_hbar(*units):
  if len(units)==0:
    hbar = hbars['au']
  else:
    try:
      assert(units[0].lower() in ['ev','cm','au'])
    except:
      raise ValueError('Not a valid units system')
    hbar = hbars[units[0].lower()]

