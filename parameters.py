
import numpy as np
from numpy import pi as PI
from numpy import sin as sin
from numpy import cos as cos

import math
from math import sqrt as sqrt

from sympy import S

import sys


# -------------------------------------------
#           Importing parameters
# -------------------------------------------

print("Importing variables")
print(sys.argv)

cline_params = sys.argv[1:]

Ntot = int(cline_params[0])
iters = int(cline_params[1])
tilt = float(cline_params[2])
deph = float(cline_params[3])
fill = int(cline_params[4])
anis = float(cline_params[5])
fdoublons = float(cline_params[6])



# -------------------------------------------
#           System parameters
# -------------------------------------------

"""
Units:
    Lengths in micrometers
    Frequencies in 1/microsecond
    Decay rate in 1/microsecond
"""
rescale = True     # Rescale frequencies with Gamma and lengths with lambda0

# Nature constants
c = 2.99*10**8       # Speed of light  micrometer/microsecond

# Atom constants
Gamma = 0.04712                     # Decay rate of excited state (1/microsecond)
lambda0 = 0.689                     # Transition wave-length (micrometers)
k0 = 2*PI/lambda0                   # wave-vector of dipole transition (1/micrometer)
omega0 = c*k0         # frequency between levels A and B  (1/microsecond)


### ------------------------
###         Method
### ------------------------
method = 'MF'           # ED: exact diagonalization
                        # LowInt: low-intensity approximation
                        # MF: mean-field
#iterations = 1          # Number of iterations, each one with a different distribution of positions
iterations = iters
seed=False               # If true, fix seed for random numbers to 0.


### ------------------------
###     Geometry settings
### ------------------------
geometry = 'random'        # Choose ensemble geometry: lattice, random

# If geometry == lattice
lattice = [10,10]                   # List of number of lattice points in each direction, i.e. sites along [Nx], [Nx,Ny] or [Nx,Ny,Nz]
latsp = 0.5*lambda0                   # Lattice spacing (micrometers). For Jun's Sr experiment: latsp=0.406

# If geometry == random
sampling = 'fermi'           # Options: gauss, fermi
#Ntotal = 10                      # Total number of atoms if geometry == random
Ntotal = Ntot                      
Rf = 9 * (Ntotal/70000.)**(1/6)                          # Fermi radius in micrometers
anisotropy = anis                                          # Ratio R_x / R_{y,z} with R_y = R_z
widths = 4.5 / (70000.0/Ntotal)**(1.0/3) * np.array([1,1,1])            # Widths of spatial Gaussian distribution of atoms (for 'random' geometry)
Tscal = 0.05                                                    # Temperature T/T_Fermi for 'fermi' sampling option
cutoffdistance = 0*lambda0              # Minimal separation between two atoms in random geometry

onsite_prefactor = 0    # On-site pre-factor coming from integrals over wave-functions (micrometers^-3)

fraction_doublons = fdoublons   # Fraction of sites with two atoms. The rest will have just one atom



### ---------------------
###     Atom properties
### ---------------------
filling=fill       # Number of atoms per site
Fe=S(11)/2                   # Note: Must use symbolic numbers S() from sympy
Fg=S(9)/2
deg_e=1
if filling==1:
    deg_g=1
    start_g=9
if filling==2:
    deg_g=2
    start_g=8
#deg_e=int(2*Fe+1)          # Degeneracy of upper levels
#deg_g=int(2*Fg+1)        # Degeneracy of lower levels
start_e=10               # Levels go from -Fe+start_e until -Fe+start_e+deg_e-1
#start_g=8               # Same for g

theta_qa=0*PI            # Spherical angles of quantization axis with respect to z.
phi_qa=0*PI              # Spherical angles of quantization axis with respect to z.

zeeman_g = 0*Gamma        # Constant energy shift between |g mu> and |g mu+1> (1/microsecond)
zeeman_e = 0*Gamma        # Constant energy shift between |e mu> and |e mu+1> (1/microsecond)
dephasing = deph*Gamma         # Adds random energy detunings to excited state. The value gives the standard deviation.
# 0.00089  9/2
# 0.004  11/2

epsilon_ne = 0*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that eigenvalues of different n_e are not degenerate
epsilon_ne2 = 0*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that n_e=2 are off-resonant


### ------------------------
###     Laser properties
### ------------------------
rabi_coupling = 0*Gamma           # Proportional to electric field strength of laser and radial dipole matrix element (1/microsecond)
detuning = 0*Gamma                  # Detuning of laser from omega0

theta_k = PI/2                 # Spherical angles of k wrt axes defined by lattice
phi_k = 0*PI

pol_x = -1      # Amplitude of x polarization in k-axis system
pol_y = 0      # Amplitude of y polarization in k-axis system     e+ = -ex-iey,  e- = ex-iey


########## Managing the lasers ##########
#
# NOTE: This is currently not implemented!!!!!
#
switchtimes = [0]       # Time step where switching happens
switchlaser = [True]       # True: Switch lasers on, False: Switch lasers off

will_laser_be_used = False
for ii in range(len(switchlaser)):
    if switchlaser[ii]==True: will_laser_be_used = True



### ------------------------
###     Initial conditions
### ------------------------
cIC = 'initialstate'     # Choose initial condition:
                    # 'initialstate': All atoms in state defined by 'initialstate'

if filling==1:
    initialstate = ['g0']
if filling==2:
    initialstate = ['g0','g1']
    
initialstate_singlons = ['g0']      # Initial state of sites with only a single atom

rotate = True      # Decide if apply rotation to initial state with H_Rabi and Rabi frequencies Omet
                    # NOTE: SAME laser properties as above.
relevantClebsch = sqrt(2/11)          # 1/2,1/2, Pi: 1/sqrt(3) Sigma: sqrt(2/3) // 1,1: Pi: 1/sqrt(2) Sigma: 1/2 // 0,1: Pi,Sigma: 1
                                      # 3/2,3/2: Pi: 3/sqrt(15), 1/sqrt(15) // 9/2,9/2: 1/(3*sqrt(11)) for Pi 1/2.
                                      # 1/2,3/2: Pi: sqrt(2/3) Sigma: 1/sqrt(3)
                                      # 9/2,11/2: C(9/2,0)=sqrt(2/11), C(7/2,9/2)=3/sqrt(11)
#Omet = PI*30/100 / (2*relevantClebsch)       # Value of Omega_Pi * tau. SAME laser as above.
Omet = PI* tilt / (2*relevantClebsch)       

### ------------------------
###         Other
### ------------------------
digits_trunc = 6        # Number of digits to truncate when computing excitation manifold of eigenstates
      
### ------------------------
###         Output
### ------------------------
outfolder = './data'
#outfolder = '.'
#outfolder = '../Simulations/First_simulations'

if method == 'MF':
    which_observables = ['populations','xy']
if method == 'LowInt':
    which_observables = ['xy']
                                        # populations: output only |nn><nn|
                                        # xy: output pauli_x, pauli_y for each pair of states

output_occupations = True
output_intensity = True
output_eigenstates = False
output_stateslist = False
append = ''
extra = ''
if deph!=0: extra = extra + '_deph%g'%(dephasing/Gamma)
if fraction_doublons!=1.0: extra = extra + '_fdoublons%g'%(fraction_doublons)

append = '_%s_Rf%g_anis%g_tilt%g%s'%(sampling,Rf/lambda0,anisotropy,Omet/PI*2*relevantClebsch,extra)
#append = '_width%g_tilt%g'%(widths[0]/lambda0,Omet/PI*2*relevantClebsch)


### ------------------------
###         Solver
### ------------------------
#solver = 'exp'         # Solve dynamics by exponentiating Linblad
solver = 'ode'          # Solve system of differential equations


### ------------------------
###     Time evolution
### ------------------------
dt=0.1/Gamma
Nt=200
max_memory=0.3        # Maximal memory allowed for superoperator. If more needed, abort program.





###
### RESCALE PARAMETERS WITH GAMMA AND LAMBDA0
###

"""
Only dimensionless quantities:

Frequency = Frequency / Gamma
Length = Length / lambda0

Note that omega = c*k ---->  omega = c/(lambda0*Gamma) k , where omega and k are rescaled with Gamma and lambda0
"""

if rescale == True:
    
    c = c/(lambda0*Gamma)           # Frequency*Length
    k0 = k0*lambda0                 # 1/Length
    
    latsp = latsp/lambda0   # Length
    Rf = Rf/lambda0   # Length
    widths = [ widths[nn]/lambda0 for nn in range(len(widths)) ]    # Length
    cutoffdistance = cutoffdistance/lambda0                         # Length
    #onsite_prefactor = onsite_prefactor*lambda0**3    # 1/Length**3  ## This is dimensionless now
    
    omega0 = omega0/Gamma            # Frequency
    zeeman_g = zeeman_g/Gamma        # Frequency
    zeeman_e = zeeman_e/Gamma        # Frequency
    dephasing = dephasing/Gamma        # Frequency
    epsilon_ne = epsilon_ne/Gamma       # Frequency
    
    rabi_coupling = rabi_coupling/Gamma  # Frequency
    detuning = detuning/Gamma                  # Frequency
    
    dt=dt*Gamma
    
    Gamma=1
    lambda0=1
    



###
### CHECKS
###

if method not in ['ED','LowInt','MF']:
    print('\nERROR/parameters: method chosen not valid.\n')
    sys.exit()
    
if method in ['MF','LowInt']:
    if deg_g+deg_e not in [2,3]:
        print('\nERROR/parameters: MF and LowInt only implemented for 3-level system, filling=2 so far.\n')
        sys.exit()
    
if method=='ED':
    print('\nERROR/parameters: ED not properly implemented yet.\n')
    sys.exit()
    
if geometry not in ['lattice','random']:
    print('\nERROR/parameters: geometry chosen not valid.\n')
    sys.exit()
    
if sampling not in ['gauss','fermi']:
    print('\nERROR/parameters: sampling chosen not valid.\n')
    sys.exit()

if deg_g>2*Fg+1 or deg_e>2*Fe+1:
    print('\nERROR/parameters: number of levels larger than degeneracy of F.\n')
    sys.exit()
    
if start_e+deg_e>2*Fe+1 or start_g+deg_g>2*Fg+1:
    print('\nERROR/parameters: number chosen for start_g/e larger than allowed by Fg/Fe.\n')
    sys.exit()
    

check_numberSw = [len(switchtimes),len(switchlaser)]
if check_numberSw.count(check_numberSw[0]) != len(check_numberSw):
    print('\nERROR/parameters: Length of switch parameter arrays inconsistent.\n')
    sys.exit()


if cIC == 'initialstate':
    if len(initialstate) != filling:
        print('\nERROR/parameters: Initial state specified incorrect length for chosen filling.\n')
        sys.exit()
        
        
if epsilon_ne!=0:
    print('\nINFO/parameters: bias term epsilon_ne is nonzero.\n')
    
    







