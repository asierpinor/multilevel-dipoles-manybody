
import math
from math import sqrt as sqrt

import numpy as np
from numpy.linalg import eig
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp

import scipy as sc
from scipy import optimize
from scipy import interpolate

import mpmath as mpmath

import os

#import scipy.sparse as sp
#from scipy.linalg import expm
#from scipy.sparse.linalg import expm as sp_expm
#from scipy.sparse.linalg import eigs
#from scipy.sparse import csc_matrix
#from scipy.sparse import lil_matrix
#from scipy.special import comb
from scipy.integrate import complex_ode

from sympy.physics.quantum.cg import CG
from sympy import S

from collections import defaultdict

import sys

import parameters as param
import hilbert_space




class Dipoles_MF:
    
    """
    Multilevel atoms in lattice or random positions.
    Dynamics using mean-field or low-intensity approximation.
    
    """
    
    def __init__(self):
        
        
        ### ---------------
        ###     CONSTANTS
        ### ---------------
        
        self.dummy_constants()
        
        
        ### ---------------
        ###     GEOMETRY
        ### ---------------
        # This is not very useful for now, except for the parameter self.Ntotal        
        if param.geometry=='lattice':
            self.Nlist = param.lattice
            
        if param.geometry=='random':
            self.Nlist = [param.Ntotal];
            
        self.dim = len(self.Nlist)
        self.Ntotal=1
        for ii in range(self.dim): self.Ntotal *= self.Nlist[ii];
        
        
        
        ### ------------------------------------
        ###  ROTATE k and polarization vectors
        ### ------------------------------------
        
        # Rotation matrices for quantization axis and laser k vector
        self.R_qaxis = np.array( [[cos(param.phi_qa),-sin(param.phi_qa),0],[sin(param.phi_qa),cos(param.phi_qa),0],[0,0,1]] ) \
                        @ np.array( [[cos(param.theta_qa),0,sin(param.theta_qa)],[0,1,0],[-sin(param.theta_qa),0,cos(param.theta_qa)]] )
        self.R_kvector = np.array( [[cos(param.phi_k),-sin(param.phi_k),0],[sin(param.phi_k),cos(param.phi_k),0],[0,0,1]] ) \
                         @ np.array( [[cos(param.theta_k),0,sin(param.theta_k)],[0,1,0],[-sin(param.theta_k),0,cos(param.theta_k)]] )
        
        # Unit vectors
        self.evec = {   'x' : np.array([1,0,0]),        # x,y,z are fixed frame
                        'y' : np.array([0,1,0]),
                        'z' : np.array([0,0,1]),
                        0   : np.array([0,0,1]),        # 0,1,-1 define quantization axis frame
                        1   : np.array([-1,-1j,0])/sqrt(2),
                        -1  : np.array([1,-1j,0])/sqrt(2) }
        
        self.evec = defaultdict( lambda: np.array([0,0,0]), self.evec )     # Set all other vectors to zero
        
        # Rotate quantization axis
        self.evec[0] = (self.R_qaxis @ np.reshape(self.evec[0],(3,1)) ).reshape(3)
        self.evec[1] = (self.R_qaxis @ np.reshape(self.evec[1],(3,1)) ).reshape(3)
        self.evec[-1] = (self.R_qaxis @ np.reshape(self.evec[-1],(3,1)) ).reshape(3)
        
        
        # Rotate laser
        self.laser_kvec = ( param.k0 * self.R_kvector @ self.evec['z'].reshape((3,1)) ).reshape(3)        # Laser wave-vector
        pol_norm = sqrt( abs(param.pol_x)**2 + abs(param.pol_y)**2 )        # Normalize polarization of laser
        self.laser_pol = ( self.R_kvector @ ( param.pol_x*self.evec['x'] + param.pol_y*self.evec['y'] ).reshape((3,1)) / pol_norm ).reshape(3)     # Laser polarization
        
        
        
        
        ### ---------------
        ###     LEVELS
        ### ---------------
        self.levels_info = { 'Fe':param.Fe, 'deg_e':param.deg_e, 'start_e':param.start_e, 'Fg':param.Fg, 'deg_g':param.deg_g, 'start_g':param.start_g }
        self.eg_to_level = { 'g': [ bb for bb in range(param.deg_g) ] ,\
                             'e': [ param.deg_g+aa for aa in range(param.deg_e) ] }
        self.level_to_eg = {}       # inverse of eg_to_level
        for ll in self.eg_to_level:
            for aa in self.eg_to_level[ll]: self.level_to_eg[aa] = ll
        
        self.Mgs = [ -param.Fg+param.start_g+mm for mm in range(param.deg_g) ]
        self.Mes = [ -param.Fe+param.start_e+mm for mm in range(param.deg_e) ]
        self.Ms = { 'g': self.Mgs, 'e': self.Mes }
        self.Qs = [0,1,-1]
        #[ expression for item in list if conditional ]
        
        
        
        ### -------------------------
        ###    CLEBSCH-GORDAN COEFF
        ### -------------------------
        self.cg = {}
        for mg in self.Mgs:
            for q in self.Qs:
                self.cg[(mg,q)] = float(CG(param.Fg, mg, 1, q, param.Fe, mg+q).doit())
        if param.Fg==0 and param.Fe==0: self.cg[(0,0)] = 1.0
        self.cg = defaultdict( lambda: 0, self.cg )     # Set all other Clebsch-Gordan coefficients to zero
        
        #print(self.cg)
        
        
        
        ### ------------------------------
        ###    SINGLE-SITE HILBERT SPACE
        ### ------------------------------
        self.hspace = hilbert_space.Hilbert_Space(self.numlevels, param.filling, 1, 'full')
        
        if param.fraction_doublons!=1.0:
            filling_singlon = 1
            self.hspace_singlon = hilbert_space.Hilbert_Space(self.numlevels, filling_singlon, 1, 'full')
        
        
        
        ### ------------------------------
        ###    SEED for RANDOM NUMBERS
        ### ------------------------------
        if param.seed:
            np.random.seed(0)
        
        
        ### ---------------
        ###     MEMORY
        ### ---------------
        if param.method == 'MF':
            self.memory_variables = ( (self.numlevels**2 + 2 )  * self.Ntotal ) * 2 * 8 / 1024**3       # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
                                                                                                        # Includes dummy variables
        if param.method == 'LowInt':
            self.memory_variables = ( (param.deg_e * param.deg_g + 1) * self.Ntotal ) * 2 * 8 / 1024**3       # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb  
        
        self.memory_Gmatrix = 2 * (self.Ntotal**2 * 9) * 2 * 8 / 1024**3
        self.memory_spacevectors = 3 * self.Ntotal * 8 / 1024**3
        self.memory = self.memory_variables + self.memory_Gmatrix + self.memory_spacevectors
        print("\nMemory: %g Gb."%(self.memory))
        
        
        
        ### ---------------------
        ###     Fill out arrays
        ### ---------------------
        if self.memory<param.max_memory:
        
            #self.fill_ri()
            #self.fill_rabi()
            self.numbering()
            #self.fill_G_matrices()
        
        else:
            print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()
        
        
        
    
    
    def dummy_constants (self):
        
        self.numlevels = param.deg_e + param.deg_g      # Total number of internal electronic states per atom
        self.cumdist_was_saved = False                         # Whether cumulative fermi distribution has been computed and saved
    
    
    def fill_position_dependent_arrays (self):
        """
        Sample positions and fill arrays that depend on positions.
        """
        
        if self.memory<param.max_memory:
        
            self.fill_ri()
            self.fill_rabi()
            self.fill_G_matrices()
            self.fill_dephasings()
        
        else:
            print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()
        
    
    
####################################################################

######                ATOM POSITIONS and RABI                #######

####################################################################
    
    def fill_ri (self):
        """Fill out matrix of atom positions r_i"""
        
        self.r_i = [np.zeros(3) for _ in range(self.Ntotal) ]
        
        if (param.geometry=='lattice'):
            """Lattice oriented along x in 1D, along xy in 2D, and along xyz in 3D
            NOTE: Not using periodic boundary conditions."""
            for r1 in range(self.Ntotal):
                ind1 = self.get_indices(r1)
                while len(ind1)<3: ind1.append(0)
                self.r_i[r1] = np.array(ind1) * param.latsp
               
        if (param.geometry=='random'):
            
            if param.sampling=='gauss':
            
                means = [0 for _ in range(len(param.widths))]
                variances = param.widths
            
                # Assign random positions to each atom and check that no other atom is too close
                for ii in range(self.Ntotal):
                
                    posvector = np.random.normal(means,variances)
                    while len(posvector)<3: posvector = np.append(posvector,0)
                
                    if param.cutoffdistance>0 and ii>0:
                    
                        count = 0
                        while not all( [ np.linalg.norm(posvector-self.r_i[jj]) > param.cutoffdistance for jj in range(ii)] ) and count<100:
                            posvector = np.random.normal(means,variances)
                            while len(posvector)<3: posvector = np.append(posvector,0)
                            count += 1
                        
                        if count>=100: print("\nWARNING/fill_ri: Atom couldn't be placed outside blockade radious\n")
                        
                    self.r_i[ii] = posvector
                    
                #print(self.r_i)
                    
                    
            if param.sampling=='fermi':
                
                if not self.cumdist_was_saved:
                    print("Start computing cum dist")
                    self.cumdistrib = self.compute_fermi_cumulative_distribution(param.Tscal)
                    self.cumdist_was_saved = True
                
                # Sampling point on sphere
                def sample_normals():
                    vec = np.random.randn(3)
                    vec /= np.linalg.norm(vec, axis=0)
                    return vec

                # Sample from cumulative distribution
                def sample_TF():
                    n = sample_normals()
                    ran1 = np.random.rand()    
                    ri=self.cumdistrib(ran1)
    
                    p = n*ri
                    return(p)
                
                # Sample random positions for each atom from fermi distribution and check that no other atom is too close
                print("Sampling atom positions")
                for ii in range(self.Ntotal):
                
                    posvector = sample_TF()
            
                    if param.cutoffdistance>0 and ii>0:
                
                        count = 0
                        while not all( [ np.linalg.norm(posvector-self.r_i[jj]) > param.cutoffdistance for jj in range(ii)] ) and count<100:
                            posvector = sample_TF()
                            count += 1
                    
                        if count>=100: print("\nWARNING/fill_ri: Atom couldn't be placed outside blockade radious\n")
                    
                    anisotropy_rescaling = np.array([ param.anisotropy**(2/3), param.anisotropy**(-1/3), param.anisotropy**(-1/3) ])
                    self.r_i[ii] = param.Rf * posvector * anisotropy_rescaling
                    
                #print(self.r_i)
                
            
            # Select sites that will contain doublons. The rest will contain singlons.
            if param.fraction_doublons != 1:
                
                # Array containing 1 for sites with doublon, 0 otherwise
                self.doublon = np.zeros(self.Ntotal, dtype=int)
                
                # Find indices of atoms with smallest distance from center (remove anisotropy first)
                distances_from_center = [ np.linalg.norm(self.r_i[jj]/anisotropy_rescaling) for jj in range(len(self.r_i))]
                
                self.doublon_sites = np.argsort( distances_from_center )
                self.doublon_sites = self.doublon_sites[ : math.floor(self.Ntotal*param.fraction_doublons) ]
                self.singlon_sites = np.delete( np.arange(self.Ntotal) , self.doublon_sites)
                
                self.doublon[self.doublon_sites] = 1
                
                print(distances_from_center)
                print(self.doublon_sites)
                print(self.singlon_sites)
                print(self.doublon)
                
                
                
                
                
                
            
                
    def get_array_position (self,indices):
        """Returns array position in row-order of a given lattice site (i1,i2,i3,...,in)"""
        if len(indices)!=self.dim: print("\nERROR/get_array_position: indices given to get_array_position have wrong length\n")
        
        array_position = indices[0]
        for ii in range(1,self.dim):
            array_position = array_position*self.Nlist[ii] + indices[ii]
        return array_position
    
    
    def get_indices (self,n):
        """Returns lattice indices (i1,i2,...) for a given array position"""
        indices = []
        temp = 0
        rest = n
        block = self.Ntotal
    
        while temp<self.dim:
            block = int( block/self.Nlist[temp] )
            indices.append(rest//block)     # should be able to do this more efficiently
            rest -= indices[temp]*block
            temp += 1
            
        return indices
    
    
    def fill_rabi (self):
        """Compute and save rabi coupling for each atom"""
        self.rabi = {}
        
        for aa in self.Mes:
            for bb in self.Mgs:
                
                coupling = param.rabi_coupling * self.cg[(bb,aa-bb)] * np.dot( self.laser_pol, np.conj(self.evec[aa-bb]) )
                
                self.rabi[(aa,bb)] = coupling * np.array([ exp( 1j * np.dot(self.laser_kvec, self.r_i[ii]) ) for ii in range(self.Ntotal) ], dtype=complex)
        
        
        #print("Rabi:")
        #print(self.rabi)
    
    def fill_dephasings (self):
        """Compute and save rabi coupling for each atom"""
        self.dephasing = np.random.normal(0,param.dephasing,self.Ntotal)
        
        #print(self.dephasing)
        
        
    def compute_fermi_cumulative_distribution (self,Tscal):
        """
        Computes the cumulative distribution of a non-interacting Fermi gas in a harmonic trap.
        Adapted from code by Thomas Bilitewski.
        """
        
        # Inverse temperature
        betascal=2./Tscal
        
        #######
        ####### Compute fugacity
        #######
        
        # Nzfun gives integral of Fermi-Dirac (non-interacting) and setting it equal to N. Then invert to get fugacity.
        def Nzfun(z):
            res=float(mpmath.polylog(3,-z))
            return(res+1./(6.*np.power(Tscal,3.)))

        # r-distribution (radius) after integrating p.
        def polylog_vec(x):
            return float(mpmath.polylog(3/2,x).real)

        vpoly = np.vectorize(polylog_vec)
        
        # Invert to get fugacity
        soli = optimize.root_scalar(Nzfun,bracket=[0,np.exp(20)])
        z=soli.root
        
        #######
        ####### Compute cumulative distribution
        #######
        
        # Set up grid
        gridscale=1.0;
        Rrho = 2*gridscale
        rho = np.linspace(0, Rrho,int(1001*gridscale))
        drho=rho[1]-rho[0];
        
        #Finite Temperature TF. test = real space distribution
        xdistrib=-vpoly(-z*np.exp(-betascal*rho**2/2))
        ntest = np.sum(xdistrib*rho**2)*drho*4.*np.pi
        xdistrib = 1./48*1./ntest *xdistrib
        
        # Radial distribution. Add volume r^2 factor
        r=rho
        dr=r[1]-r[0]
        rdens = r**2*xdistrib
        
        # Compute cumulative distribution to sample from 0 to 1 uniformly
        crdens = np.cumsum(rdens)/np.sum(rdens)
        
        # Interpolating cumulative distribution
        f = sc.interpolate.interp1d(crdens,r)
        
        return f
        
    

    
    
####################################################################

#########                ARRAY NUMBERING                ############

####################################################################


    def numbering (self):
        """
        Index all variables that will be used in MF or LowInt.
        """
        self.index = {}     # master dictionary containing all indices
        
        if param.method in ['MF']:
            self.gg_index = {}
            self.ee_index = {}
            self.eg_index = {}
            self.ge_index = {}  # auxiliary
            
            self.numbering_MF()
            self.auxiliary_numbering_MF()
            
        if param.method in ['LowInt']:
            self.eg_index = {}
            
            self.numbering_LowInt()
        
            
        #print(self.index)   
        print("Number of phase space variables: %i"%(self.nvariables))
        
        
        
    
    def numbering_MF (self):
        """
        Create dictionary where each key corresponds to a type of one-point function, e.g. sigma_{g0g1}.
        The value assigned to each key is as list of the array positions for the sigma^k_{g0g1} functions running from k=0 to k=Ntotal-1
        """
        
        if param.filling==1 or (param.filling==2 and self.numlevels==3):
            self.numbering_MF_standard()
            
        else:
            print('\nERROR/numbering_MF: Numbering function not implemented for this choice of filling and levels.')
            
            
    def numbering_MF_standard (self):
        """
        Create dictionary label -> number
        gg, ge and ee one-point functions for each 
        """
        ind = 0
        for aa in self.Mgs:
            for bb in self.Mgs:
                self.gg_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                self.ee_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mes:
            for bb in self.Mgs:
                self.eg_index[(aa,bb)] = ind
                ind += 1
        
        # Fill dummy variables. NOTE: This is a lot of variables eventually... Maybe less memory consuming way?
        dummy = ind
        ind += 1
        
        ### Make default dictionaries
        self.gg_index = defaultdict( lambda: dummy, self.gg_index )
        self.ee_index = defaultdict( lambda: dummy, self.ee_index )
        self.eg_index = defaultdict( lambda: dummy, self.eg_index )
        
        self.nvariables = ind

        
        ### Fill out master index dictionary
        self.index['gg'] = self.gg_index
        self.index['ee'] = self.ee_index
        self.index['eg'] = self.eg_index
    
    
    
    
    def auxiliary_numbering_MF (self):
        """
        Create dictionary of indices (label->number) for auxiliary variable sigma_eg, starting from last index of actual variables
        NOTE: SHOULD I AUTOMATIZE THIS??? Alternative: dictionary of dictionaries, run over ['g','e'] (MAYBE EASIER TO CHANGE LATER, e.g. REMOVE VARIABLES)
        """
        
        if param.filling==1 or (param.filling==2 and self.numlevels==3):
            self.auxiliary_numbering_MF_standard()
            
        else:
            print('\nERROR/numbering_MF: Auxiliary numbering function not implemented for this choice of filling and levels.')
    
          
    def auxiliary_numbering_MF_standard (self):
        """
        Create dictionary of indices (label->number) for auxiliary variable sigma_eg, starting from last index of actual variables
        """
        ind = self.nvariables
        
        # 1 pt extra
        self.ge_index = {}
        
        # 1 pt extra
        for aa in self.Mgs:
            for bb in self.Mes:
                self.ge_index[(aa,bb)] = ind
                ind += 1
        
        
        # Fill dummy variable
        dummy = ind
        ind += 1
        
        # Total number of auxiliary variables
        self.n_auxvariables = ind - self.nvariables
        
        # Make default dictionaries
        self.ge_index = defaultdict( lambda: dummy, self.ge_index )
        
        # Fill out master index dictionary
        self.index['ge'] = self.ge_index
        
        
        
        
    def numbering_LowInt (self):
        """
        Create dictionary where each key corresponds to a type of one-point function, e.g. sigma_{g0g1}.
        The value assigned to each key is as list of the array positions for the sigma^k_{g0g1} functions running from k=0 to k=Ntotal-1
        """
        
        if param.filling==1 or (param.filling==2 and self.numlevels==3):
            self.numbering_LowInt_standard()
            
        else:
            print('\nERROR/numbering_LowInt: Numbering function not implemented for this choice of filling and levels.')
                 
        
    def numbering_LowInt_standard (self):
        """
        Create dictionary label -> number
        eg one-point functions
        """
        ind = 0
        for aa in self.Mes:
            for bb in self.Mgs:
                self.eg_index[(aa,bb)] = ind
                ind += 1
        
        # Fill dummy variables. NOTE: This is a lot of variables eventually... Maybe less memory consuming way?
        dummy = ind
        ind += 1
        
        # Make default dictionaries
        self.eg_index = defaultdict( lambda: dummy, self.eg_index )
        
        # Number of variables
        self.nvariables = ind
        
        # Fill out master index dictionary
        self.index['eg'] = self.eg_index
        
    
    
    
    def kron_del(self,a1,a2):
        """Kroenecker delta"""
        if a1==a2:
            return 1
        else:
            return 0
    




####################################################################

###########                INTERACTIONS                #############

####################################################################


    def fill_G_matrices (self):
        """
        Fill and save interaction matrices G and Gtilde
        """
        self.G_matrix = {}
        self.Gtilde_matrix = {}
        
        # Fill G_ij depending on geometry
        Gtensor = np.zeros((self.Ntotal,self.Ntotal,3,3),dtype=complex)
        
        if param.geometry in ['lattice','random']:
            
            one = np.identity(3)
            k = param.k0
            
            print("Computing G tensor")
            for ii in range(self.Ntotal):
                """
                for jj in range(ii,self.Ntotal):
                
                    if ii==jj:
                    
                        Gtensor[ii,jj] = 1j * param.Gamma/2 * one
                    
                    else:
                        
                        # This should be vectorizable!
                        # pru2[:,:,None]*pru1[:,None,:]
                        
                        rvec = self.r_i[ii]-self.r_i[jj]
                        r = np.linalg.norm( rvec )
                        er = rvec/r
                    
                        Gtensor[ii,jj] = 3*param.Gamma/4 * exp(1j*k*r) * ( ( one - np.outer(er,er) ) * 1/(k*r) \
                                                                        + ( one - 3 * np.outer(er,er) ) * ( 1j/(k*r)**2 - 1/(k*r)**3 ) )
                                                                        
                #"""
                
                #"""
                Gtensor[ii,ii] = 1j * param.Gamma/2 * one                                                        
                
                if ii+1<self.Ntotal:
                    
                    rvec = (self.r_i[ii]).reshape(1,3)-self.r_i[ii+1:]
                    r = np.linalg.norm( rvec , axis=1 , keepdims=True )
                    er = rvec/r
                    
                    erer = er[:,:,None] * er[:,None,:]
                    r = r[:,:,None]
                    
                    Gtensor[ii,ii+1:] = 3*param.Gamma/4 * exp(1j*k*r) * ( ( one.reshape(1,3,3) - erer ) * 1/(k*r) \
                                                                    + ( one.reshape(1,3,3) - 3 * erer ) * ( 1j/(k*r)**2 - 1/(k*r)**3 ) )
                #"""
            
            print("Transposing G tensor")      
            for ii in range(self.Ntotal):
                for jj in range(ii):
                    
                    Gtensor[ii,jj] = Gtensor[jj,ii]
                    
        #print(Gtensor)
        
            
        # Multiply with polarization vectors
        print("Computing G and Gtilde matrices")
        for q1 in self.Qs:
            for q2 in self.Qs:
                
                #self.G_matrix[(q1,q2)] = np.array( [ [ self.evec[q1].conj() @ Gtensor[ii,jj] @ self.evec[q2] for jj in range(self.Ntotal) ] for ii in range(self.Ntotal) ] )
                self.G_matrix[(q1,q2)] = np.array( [ self.evec[q1].conj() @ Gtensor[ii,:] @ self.evec[q2] for ii in range(self.Ntotal) ] )
                
                #self.Gtilde_matrix[(q1,q2)] = np.array( [ [ self.evec[q1].conj() @ Gtensor[ii,jj].conj() @ self.evec[q2] for jj in range(self.Ntotal) ] for ii in range(self.Ntotal) ] )
                self.Gtilde_matrix[(q1,q2)] = np.array( [ self.evec[q1].conj() @ Gtensor[ii,:].conj() @ self.evec[q2] for ii in range(self.Ntotal) ] )
                
        
        # Default dictionary to zero if q,q' not [0,1,-1]
        self.G_matrix = defaultdict( lambda: np.zeros((self.Ntotal,self.Ntotal)) , self.G_matrix )
        self.Gtilde_matrix = defaultdict( lambda: np.zeros((self.Ntotal,self.Ntotal)) , self.Gtilde_matrix )
        
        #print(self.G_matrix)
        #print(self.Gtilde_matrix)
        
        






        

####################################################################

#######                INITIAL CONDITIONS                ###########

####################################################################
    
    def choose_initial_condition (self,cIC=param.cIC):
        """Initialize density matrix to chosen initial condition."""
        
        self.Sv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)        # Each column is a different atom
        
        # Compute single-atom rho for given IC
        self.fill_single_p_rho()
        if param.fraction_doublons!=1.0: self.fill_single_p_rho_singlon()
        
        # Compute and save initial values of expectation values
        if param.method == 'MF':
            self.fill_initial_arrays_MF()
        if param.method == 'LowInt':
            self.fill_initial_arrays_LowInt()
            
        #print(self.Sv)
        
        # Perform rotation
        if param.rotate==True:
            self.rotate_IC()
        
        #print(self.Sv)
        #print(np.concatenate( (np.arange(self.nvariables).reshape((self.nvariables,1)), self.Sv) , axis=1) )
        #print()
        
            
            
    def fill_initial_arrays_MF (self):
        """ Fill self.Sv MF array by computing expectation value wrt self.rho """
        
        for ma in self.Mgs:
            for mb in self.Mgs:
                aa = self.eg_to_level['g'][self.Mgs.index(ma)]
                bb = self.eg_to_level['g'][self.Mgs.index(mb)]
                
                # Fill doublons, assuming all sites are doublons
                expvalue = np.trace( self.hspace.sigma_matrix_local(aa,bb) @ self.rho)
                self.Sv[ self.index['gg'][(ma,mb)] ] = np.full(self.Ntotal, expvalue)
                
                # Change values of sites with singlons
                if param.fraction_doublons!=1.0:
                    expvalue = np.trace( self.hspace_singlon.sigma_matrix_local(aa,bb) @ self.rho_singlon)
                    self.Sv[ self.index['gg'][(ma,mb)] ][self.singlon_sites] = expvalue

                
        for ma in self.Mes:
            for mb in self.Mes:
                aa = self.eg_to_level['e'][self.Mes.index(ma)]
                bb = self.eg_to_level['e'][self.Mes.index(mb)]
                
                # Fill doublons, assuming all sites are doublons
                expvalue = np.trace( self.hspace.sigma_matrix_local(aa,bb) @ self.rho)
                self.Sv[ self.index['ee'][(ma,mb)] ] = np.full(self.Ntotal, expvalue)
                
                # Change values of sites with singlons
                if param.fraction_doublons!=1.0:
                    expvalue = np.trace( self.hspace_singlon.sigma_matrix_local(aa,bb) @ self.rho_singlon)
                    self.Sv[ self.index['ee'][(ma,mb)] ][self.singlon_sites] = expvalue
                
                
        for ma in self.Mes:
            for mb in self.Mgs:
                aa = self.eg_to_level['e'][self.Mes.index(ma)]
                bb = self.eg_to_level['g'][self.Mgs.index(mb)]
                
                # Fill doublons, assuming all sites are doublons
                expvalue = np.trace( self.hspace.sigma_matrix_local(aa,bb) @ self.rho)
                self.Sv[ self.index['eg'][(ma,mb)] ] = np.full(self.Ntotal, expvalue)
                
                # Change values of sites with singlons
                if param.fraction_doublons!=1.0:
                    expvalue = np.trace( self.hspace_singlon.sigma_matrix_local(aa,bb) @ self.rho_singlon)
                    self.Sv[ self.index['eg'][(ma,mb)] ][self.singlon_sites] = expvalue
                
                
                
    def fill_initial_arrays_LowInt (self):
        """ Fill self.Sv LowInt array by computing expectation value wrt self.rho """
                
        for ma in self.Mes:
            for mb in self.Mgs:
                aa = self.eg_to_level['e'][self.Mes.index(ma)]
                bb = self.eg_to_level['g'][self.Mgs.index(mb)]
                
                # Fill doublons, assuming all sites are doublons
                expvalue = np.trace( self.hspace.sigma_matrix_local(aa,bb) @ self.rho)
                self.Sv[ self.index['eg'][(ma,mb)] ] = np.full(self.Ntotal, expvalue)
                
                # Change values of sites with singlons
                if param.fraction_doublons!=1.0:
                    expvalue = np.trace( self.hspace_singlon.sigma_matrix_local(aa,bb) @ self.rho_singlon)
                    self.Sv[ self.index['eg'][(ma,mb)] ][self.singlon_sites] = expvalue
                    
        
    
    
    def rotate_IC(self):
        """Rotate initial state with driving part for time t=1."""
        
        # Define Rabi coupling for IC preparation
        self.rabi_IC = {}
        
        for aa in self.Mes:
            for bb in self.Mgs:
                
                coupling = param.Omet * self.cg[(bb,aa-bb)] * np.dot( self.laser_pol, np.conj(self.evec[aa-bb]) )
                
                self.rabi_IC[(aa,bb)] = coupling * np.array([ exp( 1j * np.dot(self.laser_kvec, self.r_i[ii]) ) for ii in range(self.Ntotal) ], dtype=complex)
                
        
        
        
        # Define mean-field equation
        def MF_eqs_IC (t,Svtemp):
            """
            Computes and returns right-hand side of mean-field equations.
            Only driving.
            """
            
            if param.filling==1:
                return MF_eqs_n1_IC(t,Svtemp)
        
            elif param.filling==2 and param.deg_g==2 and param.deg_e==1:       # Lambda-system
                return MF_eqs_n2_3Lambda_IC(t,Svtemp)
            
            else:
                print("\nERROR: EOM for this choice of level structure or filling not implemented.\n")
                return np.zeros(self.nvariables*self.Ntotal,dtype=complex)
        
        
        def MF_eqs_n1_IC (t,Svtemp):
            """Computes and returns right-hand side of mean-field equations for 1 atoms/site with arbitrary level structure.
            Only driving."""
            Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
            dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
            # gg        
            for aa in self.Mgs:
                for bb in self.Mgs:
                
                    for mm in self.Mes:
                        dSv[self.index['gg'][(aa,bb)]] += 1j * ( self.rabi_IC[(mm,aa)] * Sv[self.index['eg'][(mm,bb)]] \
                                                            - np.conj( self.rabi_IC[(mm,bb)] * Sv[self.index['eg'][(mm,aa)]] ) )
                                                            
            # ee
            for aa in self.Mes:
                for bb in self.Mes:
        
                    for nn in self.Mgs:
                        dSv[self.index['ee'][(aa,bb)]] += 1j * ( - self.rabi_IC[(bb,nn)] * Sv[self.index['eg'][(aa,nn)]] \
                                                                + np.conj( self.rabi_IC[(aa,nn)] * Sv[self.index['eg'][(bb,nn)]] ) )
                                                                
            # eg
            for aa in self.Mes:
                for bb in self.Mgs:
                    
                    for nn in self.Mgs:
                        dSv[self.index['eg'][(aa,bb)]] += 1j * np.conj( self.rabi_IC[(aa,nn)] ) * Sv[self.index['gg'][(nn,bb)]]
                    
                    for mm in self.Mes:
                        dSv[self.index['eg'][(aa,bb)]] += -1j * np.conj( self.rabi_IC[(mm,bb)] ) * Sv[self.index['ee'][(aa,mm)]]
            
            
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
        
        
        def MF_eqs_n2_3Lambda_IC (t,Svtemp):
            """
            Computes and returns right-hand side of mean-field equations for 2 atoms/site with a Lambda-type system, deg_g==2, deg_e==1.
            Only driving.
            """
            Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
            dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
            # Magnetic number of excited state
            ee = self.Mes[0]
        
            # gg        
            for aa in self.Mgs:
                for bb in self.Mgs:
                    dSv[self.index['gg'][(aa,bb)]] += 1j * ( self.rabi_IC[(ee,aa)] * Sv[self.index['eg'][(ee,bb)]] \
                                                            - np.conj( self.rabi_IC[(ee,bb)] * Sv[self.index['eg'][(ee,aa)]] ) )
                
            # ee
            for nn in self.Mgs:
                dSv[self.index['ee'][(ee,ee)]] += 1j * ( - self.rabi_IC[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] \
                                                        + np.conj( self.rabi_IC[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] ) )
                                            
            # eg
            for bb in self.Mgs:
                dSv[self.index['eg'][(ee,bb)]] += -1j * np.conj( self.rabi_IC[(ee,bb)] ) * Sv[self.index['ee'][(ee,ee)]]
                for nn in self.Mgs:
                    dSv[self.index['eg'][(ee,bb)]] += 1j * np.conj( self.rabi_IC[(ee,nn)] ) * Sv[self.index['gg'][(nn,bb)]]
                
                
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
            
            
        # Define LowInt equations
        def LowInt_eqs_IC (t,Svtemp):
            """
            Computes and returns right-hand side of low-intensity equations.
            Only driving.
            """
            
            if param.filling==1 and param.deg_g==1:
                return LowInt_eqs_n1_singleG_IC(t,Svtemp)
        
            elif param.filling==2 and param.deg_g==2 and param.deg_e==1:       # Lambda-system
                return LowInt_eqs_n2_3Lambda_IC(t,Svtemp)
            
            else:
                print("\nERROR: EOM for this choice of level structure not implemented.\n")
                return np.zeros(self.nvariables*self.Ntotal,dtype=complex)       
        
        
        def LowInt_eqs_n1_singleG_IC (t,Svtemp):
            """Computes and returns right-hand side of low-intensity equations for 1 atom/site with a single ground state, deg_g==1.
            Only driving."""
            Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
            dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
            # Magnetic number of ground state
            gg = self.Mgs[0]
        
            # eg
            for aa in self.Mes:
                dSv[self.index['eg'][(aa,gg)]] += 1j * np.conj( self.rabi_IC[(aa,gg)] )
                
                
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
            
            
        def LowInt_eqs_n2_3Lambda_IC (t,Svtemp):
            """
            Computes and returns right-hand side of low-intensity equations for 2 atoms/site with a Lambda-type structure, deg_g==2, deg_e==1.
            Only driving.
            """
            Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
            dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
            # Magnetic number of excited state
            ee = self.Mes[0]
        
            # eg
            for bb in self.Mgs:
                dSv[self.index['eg'][(ee,bb)]] += 1j * np.conj( self.rabi_IC[(ee,bb)] )  
        
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
        
        
        
        
        
        # Set solver
        if param.method in ['MF']:
            ICsolver = complex_ode(MF_eqs_IC).set_integrator('dopri5')
        if param.method in ['LowInt']:
            ICsolver = complex_ode(LowInt_eqs_IC).set_integrator('dopri5')
        
        ICsolver.set_initial_value(self.Sv.reshape( (len(self.Sv)*len(self.Sv[0])) ), 0)
        
        # Evolve for t=1
        if ICsolver.successful():
            ICsolver.integrate(ICsolver.t+1)
            self.Sv = ICsolver.y.reshape((self.nvariables,self.Ntotal))
        else: print("\nERROR/rotate_IC: Problem with ICsolver, returns unsuccessful.\n")

    



####################################################################

##########             HILBERT SPACE FUNCTIONS             #########

####################################################################

    def fill_single_p_rho (self,cIC=param.cIC):
        """Initialize density matrix of one site to chosen initial condition."""
        
        def rho_IC_initialstate ():
            """ Initialize density matrix of one site in the state specified by param.initialstate. """
            
            if len(param.initialstate)!=param.filling: print("\nWarning: Length of initialstate doesn't match filling.\n")
            
            self.rho = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
            
            occlevels = []
            for ii in range(len(param.initialstate)):
                occlevels.append( self.eg_to_level[ param.initialstate[ii][0] ][ int(param.initialstate[ii][1:]) ] )    
            occupied_localstate = self.hspace.get_statenumber( occlevels )
            
            self.rho[occupied_localstate,occupied_localstate] = 1
            
        
        optionsIC = { 'initialstate': rho_IC_initialstate }
        optionsIC[cIC]()
        
        # Check initial trace
        if abs(np.trace(self.rho)-1)>0.000000001: print("\nWARNING/choose_initial_condition: Trace of rho is initially not 1.\n")
        print("Initial trace of rho is %g."%(np.trace(self.rho).real))
        
        #print(self.rho)
        
        
    def fill_single_p_rho_singlon (self,cIC=param.cIC):
        """Initialize density matrix of one singlon site to chosen initial condition."""
        
        def rho_IC_initialstate_singlon ():
            """ Initialize density matrix of one site in the state specified by param.initialstate_singlon. """
            
            if len(param.initialstate_singlons)!=1: print("\nWarning: Length of initialstate_singlons doesn't match filling.\n")
            
            self.rho_singlon = np.zeros((self.hspace_singlon.localhilbertsize,self.hspace_singlon.localhilbertsize))
            
            occlevels = []
            for ii in range(len(param.initialstate_singlons)):
                occlevels.append( self.eg_to_level[ param.initialstate_singlons[ii][0] ][ int(param.initialstate_singlons[ii][1:]) ] )    
            occupied_localstate = self.hspace_singlon.get_statenumber( occlevels )
            
            self.rho_singlon[occupied_localstate,occupied_localstate] = 1
            
        
        optionsIC = { 'initialstate': rho_IC_initialstate_singlon }
        optionsIC[cIC]()
        
        # Check initial trace
        if abs(np.trace(self.rho_singlon)-1)>0.000000001: print("\nWARNING/choose_initial_condition: Trace of rho is initially not 1.\n")
        print("Initial trace of rho_singlon is %g."%(np.trace(self.rho_singlon).real))
        
        
        
        

    
        
        
####################################################################

###############                EOM                ##################

####################################################################
    
    
    def MF_eqs (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations"""
        
        if param.filling==1:
            return self.MF_eqs_n1(t,Svtemp)
        
        elif param.filling==2 and param.deg_g==2 and param.deg_e==1 and param.fraction_doublons==1.0:       # Lambda-system
            return self.MF_eqs_n2_3Lambda(t,Svtemp)
            
        elif param.filling==2 and param.deg_g==2 and param.deg_e==1 and param.fraction_doublons!=1.0:       # Lambda-system with singlons
            return self.MF_eqs_n2_3Lambda_withsinglons(t,Svtemp)
        
        else:
            print("\nERROR: EOM for this choice of level structure or filling not implemented.\n")
            return np.zeros(self.nvariables*self.Ntotal,dtype=complex)
    
    
    def MF_eqs_n1 (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations for 1 atom/site with arbitrary level structure."""
        Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
        dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
        # Compute arrays of auxiliary variables and add to Sv array
        # Note: After this step, the array Sv has size (self.nvariables+self.n_auxvariables , self.iterations),
        #       rows 0 to nvariables-1 correspond to main variables, the rest are auxiliary variables
        #Sv = self.fill_auxiliary_variables_MF_standard(Sv)
        
        # gg        
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.index['gg'][(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.index['gg'][(aa,bb)]]
                
                # Driving
                for mm in self.Mes:
                    
                    dSv[self.index['gg'][(aa,bb)]] += 1j * ( self.rabi[(mm,aa)] * Sv[self.index['eg'][(mm,bb)]] \
                                                        - np.conj( self.rabi[(mm,bb)] * Sv[self.index['eg'][(mm,aa)]] ) \
                                                    )
                    
                    for mm2 in self.Mes:
                    
                        # Interactions
                        dSv[self.index['gg'][(aa,bb)]] += -1j * self.cg[(aa,mm-aa)] * self.cg[(bb,mm2-bb)] \
                                                            * ( np.diagonal(self.G_matrix[(mm-aa,mm2-bb)]) - np.diagonal(self.Gtilde_matrix[(mm-aa,mm2-bb)]) ) * Sv[self.index['ee'][(mm,mm2)]]
                
                        for nn in self.Mgs:
                    
                            dSv[self.index['gg'][(aa,bb)]] += 1j * ( \
                                                                     + self.cg[(aa,mm-aa)] * self.cg[(nn,mm2-nn)] * np.diagonal(self.G_matrix[(mm-aa,mm2-nn)]) * Sv[self.index['eg'][(mm,bb)]] * np.conj(Sv[self.index['eg'][(mm2,nn)]]) \
                                                                     - self.cg[(nn,mm-nn)] * self.cg[(bb,mm2-bb)] * np.diagonal(self.Gtilde_matrix[(mm-nn,mm2-bb)]) * Sv[self.index['eg'][(mm,nn)]] * np.conj(Sv[self.index['eg'][(mm2,aa)]]) \
                                                                     )\
                                                            - 1j * ( \
                                                                    + self.cg[(aa,mm-aa)] * self.cg[(nn,mm2-nn)] * ( self.G_matrix[(mm-aa,mm2-nn)] @ np.conj(Sv[self.index['eg'][(mm2,nn)]]) ) * Sv[self.index['eg'][(mm,bb)]] \
                                                                    - self.cg[(nn,mm-nn)] * self.cg[(bb,mm2-bb)] * ( Sv[self.index['eg'][(mm,nn)]] @ self.Gtilde_matrix[(mm-nn,mm2-bb)] ) * np.conj(Sv[self.index['eg'][(mm2,aa)]]) \
                                                                    )
                
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.index['ee'][(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.index['ee'][(aa,bb)]]
        
                for nn in self.Mgs:
            
                    # Driving
                    dSv[self.index['ee'][(aa,bb)]] += 1j * ( - self.rabi[(bb,nn)] * Sv[self.index['eg'][(aa,nn)]] \
                                                            + np.conj( self.rabi[(aa,nn)] * Sv[self.index['eg'][(bb,nn)]] ) \
                                                        )
            
                    for mm in self.Mes:
                    
                        # Interactions
                        dSv[self.index['ee'][(aa,bb)]] += 1j * ( self.cg[(nn,bb-nn)] * self.cg[(nn,mm-nn)] * np.diagonal(self.G_matrix[(bb-nn,mm-nn)]) * Sv[self.index['ee'][(aa,mm)]] \
                                                                - self.cg[(nn,mm-nn)] * self.cg[(nn,aa-nn)] * np.diagonal(self.Gtilde_matrix[(mm-nn,aa-nn)]) * Sv[self.index['ee'][(mm,bb)]] )
            
                        for nn2 in self.Mgs:
                            
                            dSv[self.index['ee'][(aa,bb)]] += - 1j * ( \
                                                                     + self.cg[(nn,bb-nn)] * self.cg[(nn2,mm-nn2)] * np.diagonal(self.G_matrix[(bb-nn,mm-nn2)]) * Sv[self.index['eg'][(aa,nn)]] * np.conj(Sv[self.index['eg'][(mm,nn2)]]) \
                                                                     - self.cg[(nn,mm-nn)] * self.cg[(nn2,aa-nn2)] * np.diagonal(self.Gtilde_matrix[(mm-nn,aa-nn2)]) * Sv[self.index['eg'][(mm,nn)]] * np.conj(Sv[self.index['eg'][(bb,nn2)]]) \
                                                                    ) \
                                                            + 1j * ( \
                                                                    + self.cg[(nn,bb-nn)] * self.cg[(nn2,mm-nn2)] * ( self.G_matrix[(bb-nn,mm-nn2)] @ np.conj(Sv[self.index['eg'][(mm,nn2)]]) ) * Sv[self.index['eg'][(aa,nn)]] \
                                                                    - self.cg[(nn,mm-nn)] * self.cg[(nn2,aa-nn2)] * ( Sv[self.index['eg'][(mm,nn)]] @ self.Gtilde_matrix[(mm-nn,aa-nn2)] ) * np.conj(Sv[self.index['eg'][(bb,nn2)]]) \
                                                                    )
                
                                            
        
        # eg
        for aa in self.Mes:
            for bb in self.Mgs:
            
                # B-field and energies
                dSv[self.index['eg'][(aa,bb)]] += 1j * ( float(aa)*param.zeeman_e - float(bb)*param.zeeman_g - param.detuning + self.dephasing ) * Sv[self.index['eg'][(aa,bb)]]
            
                # Driving
                for nn in self.Mgs:
                
                    dSv[self.index['eg'][(aa,bb)]] += 1j * np.conj( self.rabi[(aa,nn)] ) * Sv[self.index['gg'][(nn,bb)]]
                    
                for mm in self.Mes:
                
                    dSv[self.index['eg'][(aa,bb)]] += -1j * np.conj( self.rabi[(mm,bb)] ) * Sv[self.index['ee'][(aa,mm)]]
            
                # Interactions
                for mm in self.Mes:
                    for nn in self.Mgs:
                    
                        dSv[self.index['eg'][(aa,bb)]] += -1j * self.cg[(nn,mm-nn)] * self.cg[(nn,aa-nn)] * np.diagonal(self.Gtilde_matrix[(mm-nn,aa-nn)]) * Sv[self.index['eg'][(mm,bb)]]
            
                        for nn2 in self.Mgs:
                
                            dSv[self.index['eg'][(aa,bb)]] += 1j * ( \
                                                                + self.cg[(nn,mm-nn)] * self.cg[(nn2,aa-nn2)] * np.diagonal(self.Gtilde_matrix[(mm-nn,aa-nn2)]) * Sv[self.index['eg'][(mm,nn)]] * Sv[self.index['gg'][(nn2,bb)]] \
                                                                )\
                                                            - 1j * ( \
                                                                + self.cg[(nn,mm-nn)] * self.cg[(nn2,aa-nn2)] * ( Sv[self.index['eg'][(mm,nn)]] @ self.Gtilde_matrix[(mm-nn,aa-nn2)] ) * Sv[self.index['gg'][(nn2,bb)]] \
                                                                )
                                                                
                        for mm2 in self.Mes:
                
                            dSv[self.index['eg'][(aa,bb)]] += 1j * ( \
                                                                - self.cg[(nn,mm-nn)] * self.cg[(bb,mm2-bb)] * np.diagonal(self.Gtilde_matrix[(mm-nn,mm2-bb)]) * Sv[self.index['eg'][(mm,nn)]] * Sv[self.index['ee'][(aa,mm2)]] \
                                                                )\
                                                            - 1j * ( \
                                                                - self.cg[(nn,mm-nn)] * self.cg[(bb,mm2-bb)] * ( Sv[self.index['eg'][(mm,nn)]] @ self.Gtilde_matrix[(mm-nn,mm2-bb)] ) * Sv[self.index['ee'][(aa,mm2)]] \
                                                                )
                
                
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
         
    
    def MF_eqs_n2_3Lambda (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations for 2 atoms/site with a Lambda-type system, deg_g==2, deg_e==1."""
        Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
        dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
        # Compute arrays of auxiliary variables and add to Sv array
        # Note: After this step, the array Sv has size (self.nvariables+self.n_auxvariables , self.iterations),
        #       rows 0 to nvariables-1 correspond to main variables, the rest are auxiliary variables
        #Sv = self.fill_auxiliary_variables_MF_standard(Sv)
        
        # Magnetic number of excited state
        ee = self.Mes[0]
        
        # gg        
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.index['gg'][(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.index['gg'][(aa,bb)]]
                
                # Driving
                dSv[self.index['gg'][(aa,bb)]] += 1j * ( self.rabi[(ee,aa)] * Sv[self.index['eg'][(ee,bb)]] \
                                                        - np.conj( self.rabi[(ee,bb)] * Sv[self.index['eg'][(ee,aa)]] ) \
                                                    )
                
                # Interactions
                dSv[self.index['gg'][(aa,bb)]] += -1j * self.cg[(aa,ee-aa)] * self.cg[(bb,ee-bb)] * ( np.diagonal(self.G_matrix[(ee-aa,ee-bb)]) - np.diagonal(self.Gtilde_matrix[(ee-aa,ee-bb)]) )
                
                for nn in self.Mgs:
                    
                    dSv[self.index['gg'][(aa,bb)]] += -1j * ( \
                                                             - self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * np.diagonal(self.G_matrix[(ee-aa,ee-nn)]) * Sv[self.index['gg'][(nn,bb)]] \
                                                             + self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['gg'][(aa,nn)]] \
                                                             ) \
                                                    + 1j * ( \
                                                             + self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * np.diagonal(self.G_matrix[(ee-aa,ee-nn)]) * Sv[self.index['eg'][(ee,bb)]] * np.conj(Sv[self.index['eg'][(ee,nn)]]) \
                                                             - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,aa)]]) \
                                                             )\
                                                    - 1j * ( \
                                                            + self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * ( self.G_matrix[(ee-aa,ee-nn)] @ np.conj(Sv[self.index['eg'][(ee,nn)]]) ) * Sv[self.index['eg'][(ee,bb)]] \
                                                            - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-bb)] ) * np.conj(Sv[self.index['eg'][(ee,aa)]]) \
                                                            )
                
                
        # ee
        for nn in self.Mgs:
                
            # B-field and energies
            dSv[self.index['ee'][(ee,ee)]] += 0
            
            # Driving
            dSv[self.index['ee'][(ee,ee)]] += 1j * ( - self.rabi[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] \
                                                    + np.conj( self.rabi[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] ) \
                                                )
            
            # Interactions
            dSv[self.index['ee'][(ee,ee)]] += 1j * self.cg[(nn,ee-nn)] * self.cg[(nn,ee-nn)] * ( np.diagonal(self.G_matrix[(ee-nn,ee-nn)]) - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-nn)]) )
            
            for mm in self.Mgs:
                
                dSv[self.index['ee'][(ee,ee)]] += self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * (\
                                                - 1j * ( np.diagonal(self.G_matrix[(ee-nn,ee-mm)]) - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) ) * Sv[self.index['gg'][(mm,nn)]] \
                                                - 1j * ( np.diagonal(self.G_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,mm)]]) \
                                                         - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,mm)]]) ) \
                                                + 1j * ( ( self.G_matrix[(ee-nn,ee-mm)] @ np.conj(Sv[self.index['eg'][(ee,mm)]]) ) * Sv[self.index['eg'][(ee,nn)]] \
                                                        - ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-mm)] ) * np.conj(Sv[self.index['eg'][(ee,mm)]]) )\
                                                )
                
                                            
        
        # eg
        for bb in self.Mgs:
            
            # B-field and energies
            dSv[self.index['eg'][(ee,bb)]] += 1j * ( - float(bb)*param.zeeman_g - param.detuning + self.dephasing ) * Sv[self.index['eg'][(ee,bb)]]
            
            # Driving
            dSv[self.index['eg'][(ee,bb)]] += -1j * np.conj( self.rabi[(ee,bb)] ) * Sv[self.index['ee'][(ee,ee)]]
            
            for nn in self.Mgs:
                
                dSv[self.index['eg'][(ee,bb)]] += 1j * np.conj( self.rabi[(ee,nn)] ) * Sv[self.index['gg'][(nn,bb)]]
            
                # Interactions
                dSv[self.index['eg'][(ee,bb)]] += -1j * self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]]
            
                for mm in self.Mgs:
                
                    dSv[self.index['eg'][(ee,bb)]] += 1j * ( \
                                                        + self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * Sv[self.index['gg'][(mm,bb)]] \
                                                        - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]] * Sv[self.index['ee'][(ee,ee)]] \
                                                        )\
                                                    - 1j * ( \
                                                        + self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-mm)] ) * Sv[self.index['gg'][(mm,bb)]] \
                                                        - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-bb)] ) * Sv[self.index['ee'][(ee,ee)]] \
                                                        )
                
                
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
        
    
    def MF_eqs_n2_3Lambda_withsinglons (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations for 2 atoms/site with a Lambda-type system, deg_g==2, deg_e==1."""
        Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
        dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
        # Compute arrays of auxiliary variables and add to Sv array
        # Note: After this step, the array Sv has size (self.nvariables+self.n_auxvariables , self.iterations),
        #       rows 0 to nvariables-1 correspond to main variables, the rest are auxiliary variables
        #Sv = self.fill_auxiliary_variables_MF_standard(Sv)
        
        # Magnetic number of excited state
        ee = self.Mes[0]
        
        # gg        
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.index['gg'][(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.index['gg'][(aa,bb)]]
                
                # Driving
                dSv[self.index['gg'][(aa,bb)]] += 1j * ( self.rabi[(ee,aa)] * Sv[self.index['eg'][(ee,bb)]] \
                                                        - np.conj( self.rabi[(ee,bb)] * Sv[self.index['eg'][(ee,aa)]] ) \
                                                    )
                
                # Interactions
                dSv[self.index['gg'][(aa,bb)]] += -1j * self.doublon * self.cg[(aa,ee-aa)] * self.cg[(bb,ee-bb)] * ( np.diagonal(self.G_matrix[(ee-aa,ee-bb)]) - np.diagonal(self.Gtilde_matrix[(ee-aa,ee-bb)]) )
                
                
                # Singlon self-int
                dSv[self.index['gg'][(aa,bb)]] += -1j * (1-self.doublon) * self.cg[(aa,ee-aa)] * self.cg[(bb,ee-bb)] \
                                                    * ( np.diagonal(self.G_matrix[(ee-aa,ee-bb)]) - np.diagonal(self.Gtilde_matrix[(ee-aa,ee-bb)]) ) * Sv[self.index['ee'][(ee,ee)]]
                
                for nn in self.Mgs:
                    
                    dSv[self.index['gg'][(aa,bb)]] += -1j * self.doublon * ( \
                                                             - self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * np.diagonal(self.G_matrix[(ee-aa,ee-nn)]) * Sv[self.index['gg'][(nn,bb)]] \
                                                             + self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['gg'][(aa,nn)]] \
                                                             ) \
                                                    + 1j * ( \
                                                             + self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * np.diagonal(self.G_matrix[(ee-aa,ee-nn)]) * Sv[self.index['eg'][(ee,bb)]] * np.conj(Sv[self.index['eg'][(ee,nn)]]) \
                                                             - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,aa)]]) \
                                                             )\
                                                    - 1j * ( \
                                                            + self.cg[(aa,ee-aa)] * self.cg[(nn,ee-nn)] * ( self.G_matrix[(ee-aa,ee-nn)] @ np.conj(Sv[self.index['eg'][(ee,nn)]]) ) * Sv[self.index['eg'][(ee,bb)]] \
                                                            - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-bb)] ) * np.conj(Sv[self.index['eg'][(ee,aa)]]) \
                                                            )
                
                
                
        # ee
        
        # B-field and energies
        dSv[self.index['ee'][(ee,ee)]] += 0
        
        for nn in self.Mgs:
            
            # Driving
            dSv[self.index['ee'][(ee,ee)]] += 1j * ( - self.rabi[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] \
                                                    + np.conj( self.rabi[(ee,nn)] * Sv[self.index['eg'][(ee,nn)]] ) \
                                                )
            
            # Interactions
            dSv[self.index['ee'][(ee,ee)]] += 1j * self.doublon * self.cg[(nn,ee-nn)] * self.cg[(nn,ee-nn)] * ( np.diagonal(self.G_matrix[(ee-nn,ee-nn)]) - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-nn)]) )
            
            # Singlon self-int
            dSv[self.index['ee'][(ee,ee)]] += 1j * (1-self.doublon) *\
                                                     ( self.cg[(nn,ee-nn)] * self.cg[(nn,ee-nn)] * np.diagonal(self.G_matrix[(ee-nn,ee-nn)]) * Sv[self.index['ee'][(ee,ee)]] \
                                                      - self.cg[(nn,ee-nn)] * self.cg[(nn,ee-ee)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-nn)]) * Sv[self.index['ee'][(ee,ee)]] )
            
            for mm in self.Mgs:
                
                dSv[self.index['ee'][(ee,ee)]] += self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * (\
                                                - 1j * self.doublon * ( np.diagonal(self.G_matrix[(ee-nn,ee-mm)]) - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) ) * Sv[self.index['gg'][(mm,nn)]] \
                                                - 1j * ( np.diagonal(self.G_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,mm)]]) \
                                                         - np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * np.conj(Sv[self.index['eg'][(ee,mm)]]) ) \
                                                + 1j * ( ( self.G_matrix[(ee-nn,ee-mm)] @ np.conj(Sv[self.index['eg'][(ee,mm)]]) ) * Sv[self.index['eg'][(ee,nn)]] \
                                                        - ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-mm)] ) * np.conj(Sv[self.index['eg'][(ee,mm)]]) )\
                                                )                         
                
                                            
        
        # eg
        for bb in self.Mgs:
            
            # B-field and energies
            dSv[self.index['eg'][(ee,bb)]] += 1j * ( - float(bb)*param.zeeman_g - param.detuning + self.dephasing ) * Sv[self.index['eg'][(ee,bb)]]
            
            # Driving
            dSv[self.index['eg'][(ee,bb)]] += -1j * np.conj( self.rabi[(ee,bb)] ) * Sv[self.index['ee'][(ee,ee)]]
            
            for nn in self.Mgs:
                
                dSv[self.index['eg'][(ee,bb)]] += 1j * np.conj( self.rabi[(ee,nn)] ) * Sv[self.index['gg'][(nn,bb)]]
            
                # Interactions
                dSv[self.index['eg'][(ee,bb)]] += -1j * self.doublon * self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]]
                
                # Singlon self-int
                dSv[self.index['eg'][(ee,bb)]] += -1j * (1-self.doublon) * self.cg[(nn,ee-nn)] * self.cg[(nn,ee-nn)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-nn)]) * Sv[self.index['eg'][(ee,bb)]]
            
                for mm in self.Mgs:
                
                    dSv[self.index['eg'][(ee,bb)]] += 1j * ( \
                                                        + self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-mm)]) * Sv[self.index['eg'][(ee,nn)]] * Sv[self.index['gg'][(mm,bb)]] \
                                                        - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * np.diagonal(self.Gtilde_matrix[(ee-nn,ee-bb)]) * Sv[self.index['eg'][(ee,nn)]] * Sv[self.index['ee'][(ee,ee)]] \
                                                        )\
                                                    - 1j * ( \
                                                        + self.cg[(nn,ee-nn)] * self.cg[(mm,ee-mm)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-mm)] ) * Sv[self.index['gg'][(mm,bb)]] \
                                                        - self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * ( Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-bb)] ) * Sv[self.index['ee'][(ee,ee)]] \
                                                        )
                                                        
                
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
    
    
    
    
    def LowInt_eqs (self,t,Svtemp):
        """Computes and returns right-hand side of low-intensity equations"""
        
        if param.filling==1 and param.deg_g==1:
            return self.LowInt_eqs_n1_singleG(t,Svtemp)
        
        elif param.filling==2 and param.deg_g==2 and param.deg_e==1 and param.fraction_doublons==1.0:       # Lambda-system
            return self.LowInt_eqs_n2_3Lambda(t,Svtemp)
            
        else:
            print("\nERROR: EOM for this choice of level structure not implemented.\n")
            return np.zeros(self.nvariables*self.Ntotal,dtype=complex)       
              
              
    def LowInt_eqs_n1_singleG (self,t,Svtemp):
        """Computes and returns right-hand side of LowInt equations for 1 atom/site with a single ground state, deg_g==1."""
        Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
        dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
        # Magnetic number of ground state
        gg = self.Mgs[0]
        
        # eg
        for aa in self.Mes:
        
            # B-field and energies
            dSv[self.index['eg'][(aa,gg)]] += 1j * ( float(aa)*param.zeeman_e - param.detuning + self.dephasing ) * Sv[self.index['eg'][(aa,gg)]]
        
            # Driving
            dSv[self.index['eg'][(aa,gg)]] += 1j * np.conj( self.rabi[(aa,gg)] )
        
            # Interactions
            for mm in self.Mes:
        
                dSv[self.index['eg'][(aa,gg)]] += - 1j * self.cg[(gg,mm-gg)] * self.cg[(gg,aa-gg)] * ( Sv[self.index['eg'][(mm,gg)]] @ self.Gtilde_matrix[(mm-gg,aa-gg)] )
                
                
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
          
        
    def LowInt_eqs_n2_3Lambda (self,t,Svtemp):
        """Computes and returns right-hand side of low-intensity equations for 2 atoms/site with a Lambda-type structure, deg_g==2, deg_e==1."""
        Sv = Svtemp.reshape( (self.nvariables,self.Ntotal) )
        dSv = np.zeros((self.nvariables,self.Ntotal),dtype=complex)
        
        # Magnetic number of excited state
        ee = self.Mes[0]
        
        # eg
        for bb in self.Mgs:
            
            # B-field and energies
            dSv[self.index['eg'][(ee,bb)]] += 1j * ( - float(bb)*param.zeeman_g - param.detuning + self.dephasing ) * Sv[self.index['eg'][(ee,bb)]]
            
            # Driving
            dSv[self.index['eg'][(ee,bb)]] += 1j * np.conj( self.rabi[(ee,bb)] )
            
            # Interactions
            for nn in self.Mgs:
                
                dSv[self.index['eg'][(ee,bb)]] += - 1j * self.cg[(nn,ee-nn)] * self.cg[(bb,ee-bb)] * Sv[self.index['eg'][(ee,nn)]] @ self.Gtilde_matrix[(ee-nn,ee-bb)]    
                
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
    
    
    
    
    def fill_auxiliary_variables_MF_standard (self,mainSv):
        """
        Return array with auxiliary variables computed from array Sv of variables. In this case just sigma_ge.
        """
        Sv = np.concatenate( ( mainSv , np.zeros((self.n_auxvariables,self.Ntotal),dtype=complex) ), axis=0)
        
        # 1pt extra
        for aa in self.Mgs:
            for bb in self.Mes:
                Sv[ self.index['ge'][(aa,bb)] ] = np.conj( Sv[ self.index['eg'][(bb,aa)] ] )
        
        # Return array containing main and auxiliary variables
        return Sv
    





        
        
####################################################################

############                DYNAMICS                ###############

####################################################################
       
    def set_solver (self,time):
        """ Choose solver for ODE and set initial conditions. """
        if param.method in ['MF']:
            self.solver = complex_ode(self.MF_eqs).set_integrator('dopri5')
        if param.method in ['LowInt']:
            self.solver = complex_ode(self.LowInt_eqs).set_integrator('dopri5')
        
        self.solver.set_initial_value(self.Sv.reshape( (len(self.Sv)*len(self.Sv[0])) ), time)
        
    
    def evolve_onestep (self):
        """ Evolve classical variables from t to t+dt. """
        self.solver.integrate(self.solver.t+param.dt)
        self.Sv = self.solver.y.reshape((self.nvariables,self.Ntotal))
        
        #print(self.Sv[self.ee_index[(0,0)]] + self.Sv[self.gg_index[(0,0)]])
        
        



####################################################################

#############                OUTPUT                ################

####################################################################
        
    
    def read_occs (self):
        """
        Outputs occupation of each level, summed over all atoms.
        """
        out = []
        
        # Occupations
        if 'populations' in param.which_observables:
            
            if param.method in ['MF']:
                
                for aa in self.Mgs:
                    out.append( np.sum( self.Sv[self.index['gg'][(aa,aa)]].real )/self.Ntotal )
                for bb in self.Mes:
                    out.append( np.sum( self.Sv[self.index['ee'][(bb,bb)]].real )/self.Ntotal )
                    
            if param.method in ['LowInt']:
                print("\nERROR: population can not be outputed for LowInt.\n")
            
                
        if 'xy' in param.which_observables:
            
            if param.method in ['MF']:
        
                for aa in range(self.hspace.localhilbertsize):
                    for bb in range(aa+1,self.hspace.localhilbertsize):
                    
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                    
                        coh = np.conj( self.Sv[self.index[lb+la][(mb,ma)]] )        # Write in inverse order because Sv contains 'eg', but not 'ge'
                    
                        out.append( np.sum( coh.real ) / self.Ntotal )
                        out.append( np.sum( coh.imag ) / self.Ntotal )
                        
                        
            if param.method in ['LowInt']:
                
                for aa in range(self.hspace.localhilbertsize):
                    for bb in range(aa+1,self.hspace.localhilbertsize):
                    
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                        
                        if la=='g' and lb=='e':     # Only eg coherences
                    
                            coh = np.conj( self.Sv[self.index[lb+la][(mb,ma)]] )        # Write in inverse order because Sv contains 'eg', but not 'ge'
                    
                            out.append( np.sum( coh.real ) / self.Ntotal )
                            out.append( np.sum( coh.imag ) / self.Ntotal )
                
        
        
        return out


    
    def output_intensity (self):
        """
        Outputs light intensity along x,y,z, or whatever direction is chosen.
        """
        out = []
        
        rs = { 'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z': np.array([0,0,1]) }
        output_directions = ['x','y','z']
            
        for direction in output_directions:
            
            r = rs[direction]
            er = r/np.linalg.norm(r)
            proj = np.identity(3) - np.outer(er,er)
            
            phases = np.array( [ np.exp(-1j*param.k0*np.dot(er,self.r_i[ii])) for ii in range(self.Ntotal) ] )
            
            # Compute < Efield > and add |E|^2 to intensity
            Efield = np.zeros(3,dtype=complex)
            for mm in self.Mes:
                for nn in self.Mgs:
                    Efield += self.cg[(nn,mm-nn)] * self.evec[mm-nn] * np.dot( phases, np.conj(self.Sv[self.index['eg'][(mm,nn)]]) )
            Efield = np.squeeze( np.dot(proj, Efield.reshape(3,1)) )
            
            intensity = np.dot(Efield, np.conj(Efield))
            
            
            # Remove self-interaction of mean values
            for mm in self.Mes:
                for nn in self.Mgs:
                    for mm2 in self.Mes:
                        for nn2 in self.Mgs:
                            
                            temp = self.cg[(nn,mm-nn)] * self.cg[(nn2,mm2-nn2)] * np.dot( np.dot(proj,self.evec[mm-nn]) , np.dot(proj, np.conj(self.evec[mm2-nn2])) )
                            
                            intensity -= temp * np.sum( self.Sv[self.index['eg'][(mm2,nn2)]] * np.conj(self.Sv[self.index['eg'][(mm,nn)]]) )
            
            
            # Add i==j term. (For LowInt sigma_ee is zero and is not saved.)
            if param.method in ['MF']:
                
                if param.filling==1:
                
                    for mm in self.Mes:
                        for nn in self.Mgs:
                            for mm2 in self.Mes:
                            
                                temp = self.cg[(nn,mm-nn)] * self.cg[(nn,mm2-nn)] * np.dot( np.dot(proj,self.evec[mm-nn]) , np.dot(proj, np.conj(self.evec[mm2-nn]) ) )
                            
                                intensity += temp * np.sum( self.Sv[self.index['ee'][(mm2,mm)]] )
                                
                if param.filling==2 and param.deg_g==2 and param.deg_e==1:       # Lambda-system
                
                    ee = self.Mes[0]
                    
                    for nn in self.Mgs:
                        for nn2 in self.Mgs:
                        
                            temp = self.cg[(nn,ee-nn)] * self.cg[(nn2,ee-nn2)] * np.dot( np.dot(proj,self.evec[ee-nn]) , np.dot(proj, np.conj(self.evec[ee-nn2]) ) )
                        
                            intensity += temp * ( self.Ntotal * self.kron_del(nn,nn2) - np.sum( self.Sv[self.index['gg'][(nn,nn2)]] ) ) 
            
            assert abs(intensity.imag) < 0.00000001
            
            out.append(intensity.real)
            
        return out
            
        
        
    

    '''
        
    '''
        











