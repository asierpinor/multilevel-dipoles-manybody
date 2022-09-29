# 

import numpy as np
import matplotlib.pyplot as plt
#import cmath
#from cmath import exp as exp
from numpy import pi as PI
from numpy import exp as exp
from numpy import sin as sin
from numpy import cos as cos

from scipy.optimize import curve_fit

import time
import sys


import parameters as param
import dipoles_ED
import dipoles_MF


"""
Explain code here:


"""



# -------------------------------------------
#           Functions
# -------------------------------------------
        
        
# Prints execution time in hours, minutes and seconds
def print_time(t):
    hours = t//3600;
    minutes = (t-3600*hours)//60;
    seconds = (t-3600*hours-60*minutes);
    print("Execution time: %i hours, %i minutes, %g seconds"%(hours,minutes,seconds));






# -------------------------------------------
#           Set up system
# -------------------------------------------

starttime = time.time()
lasttime = starttime

print("\n-------> Setting up system\n")


"""
if param.method=='ED':
    
    print("\nMethod: ED\n")
    dipsys = dipoles_ED.Dipoles_ED()
    
    if param.Nt>0:
        if dipsys.memory_estimate_sparse_Linblad>param.max_memory:
            print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()

    if param.output_eigenstates:
        if dipsys.memory_full_Hamiltonian>param.max_memory:
            print("Memory of full Hamiltonian is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()
"""


if param.method in ['MF','LowInt']:
    
    print("\nMethod: %s\n"%(param.method))
    dipsys = dipoles_MF.Dipoles_MF()
    



"""

# -------------------------------------------
#           Compute eigenvectors
# -------------------------------------------

print("\n-------> Computing eigenvectors\n")

if param.output_eigenstates:
    
    dipsys.define_Heff_dipolar()
    
    dipsys.compute_eigenstates()


    print("\nTime computing eigenvectors.")
    print_time(time.time()-lasttime)
    lasttime = time.time()
    
else: print("No.")


"""
    



# -------------------------------------------
#                 Dynamics
# -------------------------------------------

print("\n-------> Computing dynamics\n")

if param.Nt>0:
    
    # Set up output files

    for it in range(param.iterations):
        
        # Sample/set atom positions
        dipsys.fill_position_dependent_arrays()
        
        # Prepare output
        if param.output_occupations:
            data_expvalues_temp = []
        
        if param.output_intensity:
            data_intensity_temp = []
        

        # ----------------------
        #  Set up initial time
        # ----------------------

        phase = 0       # Traces stage of evolution, if Hamiltonian has sudden changes in time, e.g. laser switch on/off

        dipsys.choose_initial_condition()
    
        """
        if param.method=='ED':
    
            dipsys.save_partial_HamLin() # Computes ham_eff_dipoles again. Change code to avoid it.

            dipsys.define_linblad_superop(phase)

            dipsys.compute_memory()
            if dipsys.memory_sparse_Linblad+dipsys.memory_sparse_Hamiltonian > param.max_memory:
                print("Memory is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
                sys.exit()

            if param.solver == 'exp': dipsys.compute_evolution_op()
            if param.solver == 'ode': dipsys.set_solver(0)
    
            print("\nTime for constructing Linblad at t=0.")
            print_time(time.time()-lasttime)
            lasttime = time.time()
    
        """
    
        if param.method in ['MF','LowInt']:
        
            dipsys.set_solver(0)



        # -----------
        #  Save t=0
        # -----------

        times=[ 0 ]
    
    
        # Prepare output
        if param.output_occupations:
            """
            if param.method=='ED':
                dipsys.create_output_ops_occs()
            """
            data_expvalues_temp.append(dipsys.read_occs())
            #data_expvalues = [ dipsys.read_occs() ]
        
        if param.output_intensity:
            data_intensity_temp.append(dipsys.output_intensity())
            #data_intensity = [ dipsys.output_intensity() ]
        
        # -----------
        #  Evolve
        # -----------
        for tt in range(1,param.Nt+1):
    
            print(times[tt-1])
        
            # One step forward
            if param.method in ['MF','LowInt']:
                if dipsys.solver.successful():
                    dipsys.evolve_onestep()
                else: print("\nERROR: Problem with solver, returns unsuccessful.\n")
                times.append(dipsys.solver.t)
        
            """
            if param.method=='ED':
                if param.solver == 'ode':
                    if dipsys.solver.successful():
                        dipsys.evolve_rho_onestep()
                    else: print("\nERROR: Problem with solver, returns unsuccessful.\n")
                    times.append(dipsys.solver.t)
    
                if param.solver == 'exp':
                    dipsys.evolve_rho_onestep()
                    times.append( tt*param.dt )
        
            """
        
            # Save observables
            if param.output_occupations: data_expvalues_temp.append(dipsys.read_occs())
            if param.output_intensity: data_intensity_temp.append(dipsys.output_intensity())
         
            """
    
            if len(param.switchtimes)>phase+1:
                if param.switchtimes[phase+1]==tt:
                    phase = phase + 1
                    dipsys.decide_if_timeDep(phase)
                    dipsys.define_linblad_superop(phase)
                    if param.solver == 'exp': dipsys.compute_evolution_op()
                    if param.solver == 'ode': dipsys.set_solver(dipsys.solver.t)
            """
            
        """
        #print(dipsys.rho)
        print("Purity:")
        print(np.trace(dipsys.rho@dipsys.rho))
        """
            
        # Add observables to mean
        if it==0:
            data_expvalues = data_expvalues_temp
            data_intensity = data_intensity_temp
            
        else:
            data_expvalues = ( np.array(data_expvalues) + np.array(data_expvalues_temp) ).tolist()
            data_intensity = ( np.array(data_intensity) + np.array(data_intensity_temp) ).tolist()
        
        
        
        print("\nTime for evolving.")
        print_time(time.time()-lasttime)
        lasttime = time.time()
    
    
else: print("No.")
    
    

# Normalize data
data_expvalues = ( np.array(data_expvalues)/param.iterations ).tolist()
data_intensity = ( np.array(data_intensity)/param.iterations ).tolist()



# -------------------------------------------
#           Output data
# -------------------------------------------

comments = ''
comments = comments + '# method = %s\n'%(param.method)
comments = comments + '# iterations = %i\n'%(param.iterations)
comments = comments + '# seed = %s\n'%(param.seed)
comments = comments + '# \n'

comments = comments + '# geometry = %s\n'%(param.geometry)
comments = comments + '# \n# Lattice info'
comments = comments + '# Lattice = %s\n'%( ', '.join([str(param.lattice[ii]) for ii in range(len(param.lattice))]) )
comments = comments + '# Lattice spacing = %i\n'%(param.latsp)
comments = comments + '# \n# Random positions info:'
comments = comments + '# Ntotal (random) = %i\n'%(param.Ntotal)
comments = comments + '# Widths = %s\n'%( ', '.join([str(param.widths[ii]) for ii in range(len(param.widths))]) )
comments = comments + '# cutoffdistance = %i\n'%(param.cutoffdistance)
comments = comments + '# fraction_doublons = %g\n'%(param.fraction_doublons)
comments = comments + '# \n'

comments = comments + '# filling = %i\n'%(param.filling)
comments = comments + '# Fe = %g\n'%(float(param.Fe))
comments = comments + '# Fg = %g\n'%(float(param.Fg))
comments = comments + '# deg_e = %i\n'%(param.deg_e)
comments = comments + '# deg_g = %i\n'%(param.deg_g)
comments = comments + '# start_e = %i\n'%(param.start_e)
comments = comments + '# start_g = %i\n'%(param.start_g)
comments = comments + '# \n'

comments = comments + '# Gamma = %g\n'%(param.Gamma)
comments = comments + '# lambda0 = %g\n'%(param.lambda0)
comments = comments + '# theta_qa = %g\n'%(param.theta_qa)
comments = comments + '# phi_qa = %g\n'%(param.phi_qa)
comments = comments + '# zeeman_e = %g\n'%(param.zeeman_e)
comments = comments + '# zeeman_g = %g\n'%(param.zeeman_g)
comments = comments + '# dephasing = %g\n'%(param.dephasing)
comments = comments + '# epsilon_ne = %g\n'%(param.epsilon_ne)
comments = comments + '# epsilon_ne2 = %g\n'%(param.epsilon_ne2)
comments = comments + '# \n'

comments = comments + '# rabi_coupling = %g\n'%(param.rabi_coupling)
comments = comments + '# detuning = %g\n'%(param.detuning)
comments = comments + '# theta_k = %g\n'%(param.theta_k)
comments = comments + '# phi_k = %g\n'%(param.phi_k)
comments = comments + '# pol_x = %g\n'%(param.pol_x)
comments = comments + '# pol_y = %g\n'%(param.pol_y)
comments = comments + '# \n'

comments = comments + '# Laser switch times = %s\n'%( ', '.join([str(param.switchtimes[ii]) for ii in range(len(param.switchtimes))]) )
comments = comments + '# Laser on/off = %s\n'%( ', '.join([str(param.switchlaser[ii]) for ii in range(len(param.switchlaser))]) )
comments = comments + '# \n'

comments = comments + '# IC = %s\n'%(param.cIC)
comments = comments + '# initialstate = %s\n'%('| '+' '.join(param.initialstate)+' >')
comments = comments + '# initialstate_singlons = %s\n'%('| '+' '.join(param.initialstate_singlons)+' >')
comments = comments + '# rotate = %s\n'%(str(param.rotate))
comments = comments + '# Omet = %g\n'%(param.Omet)

comments = comments + '# which_observables = %s\n'%('[ '+', '.join([ str(param.which_observables[ii]) for ii in range(len(param.which_observables)) ])+' ]')

comments = comments + '# solver = %s\n'%(param.solver)
comments = comments + '# dt = %g\n'%(param.dt)
comments = comments + '# Nt = %i\n'%(param.Nt)
comments = comments + '# \n# \n'


# Initial conditions string
if param.cIC=='initialstate':
    string_IC = '_IC%s'%(''.join(param.initialstate))


# States occupations
if param.output_occupations:
    #if param.method=='ED':
    if param.method in ['MF','LowInt']:
        string_out = '%s_%s_Ntot%i_fill%i_Fg%g_Fe%g_Ng%i_Ne%i%s_iter%i%s'%(param.method, param.geometry, dipsys.Ntotal, param.filling, float(param.Fg), float(param.Fe), \
                                                                    param.deg_g, param.deg_e, string_IC, param.iterations, param.append)
    filename = '%s/expvals_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: time | %s .\n'%('[ '+', '.join([ str(param.which_observables[ii]) for ii in range(len(param.which_observables)) ])+' ]') )
        f.write('# \n# \n')
        output_data = [ [ times[tt] ] + data_expvalues[tt] for tt in range(len(times)) ]
        np.savetxt(f,output_data,fmt='%.6g')
        
        
# Output intensity
if param.output_intensity:
    #if param.method=='ED':
    if param.method in ['MF','LowInt']:
        string_out = '%s_%s_Ntot%i_fill%i_Fg%g_Fe%g_Ng%i_Ne%i%s_iter%i%s'%(param.method, param.geometry, dipsys.Ntotal, param.filling, float(param.Fg), float(param.Fe), \
                                                                    param.deg_g, param.deg_e, string_IC, param.iterations, param.append)
    filename = '%s/intensity_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: time | I_x | I_y | I_z.\n')
        f.write('# \n# \n')
        output_data = [ [ times[tt] ] + data_intensity[tt] for tt in range(len(times)) ]
        np.savetxt(f,output_data,fmt='%.6g')


print("\nTime for file output.")
print_time(time.time()-lasttime)
lasttime = time.time()

"""
    

# Eigenstates
if param.output_eigenstates:
    filename = '%s/eigenstates_%s_fill%i_Ng%i_Ne%i_Ni%i_C%g%s.txt'%(param.outfolder, param.dipole_structure, param.filling,\
                 param.deg_g, param.deg_e, param.deg_i, param.onsite_prefactor, param.append)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: energy | Col 2: decay rate | Col 3: number of excitations | Col 4: number of states involved | Col >=5: square_abs of amplitudes of eigenstate.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*2 + ['%i\t%i'] + [floatfmt]*dipsys.hspace.hilbertsize )
        output_data = [ [ dipsys.evalues[ii].real , -2*dipsys.evalues[ii].imag , dipsys.excitations_estates[ii] , dipsys.nstatesinvolved_estates[ii] ] + list(abs(dipsys.estates[:,ii])**2) for ii in range(len(dipsys.evalues)) ]
        #formatstring = '\t'.join( [floatfmt]*2 + [floatfmt]*dipsys.hspace.hilbertsize )
        #output_data = [ [ dipsys.evalues[ii].real , -2*dipsys.evalues[ii].imag ] + list(abs(dipsys.estates[:,ii])**2) for ii in range(len(dipsys.evalues)) ]
        np.savetxt(f,output_data,fmt=formatstring)
        
        
    #dipsys.read_occs_totalF(5)

    #for ii in range(len(dipsys.estates[:,6])):
        
        #print(ii, dipsys.truncate(dipsys.estates[ii,6].real,5), dipsys.truncate(dipsys.estates[ii,6].imag,5) )
        #print(dipsys.truncate(4.12452151,4))

"""


# -------------------------------------------
#           Plots
# -------------------------------------------

"""

###
### Plot individual occupancies of each state.
###
#for nn in range(dipsys.hspace.localhilbertsize):
if param.output_occupations:
    for nn in range(len(data_expvalues[0])):
        plt.plot( times, [data_expvalues[ii][nn] for ii in range(len(data_expvalues))] , label=r'$| %i \rangle \langle %i |$'%(nn,nn))
    #plt.plot( times, exp(-np.array(times)) , label='exp')
    plt.xlabel('Time: ' + r'$t$')
    plt.ylabel('State occupancies: ' + r'$| \alpha \rangle \langle \alpha |$')
    plt.xlim(0,param.Nt*param.dt)
    plt.legend(loc='upper right')
    plt.savefig('%s/states_occ_%s_fill%i_Ng%i_Ne%i_C%g_IC%s%s.pdf'%(param.outfolder, param.dipole_structure, param.filling,\
                     param.deg_g, param.deg_e, param.onsite_prefactor, ''.join(param.initialstate), param.append))

                     
print("\nTime for plotting.")
print_time(time.time()-lasttime)
lasttime = time.time()

"""



print("\nTotal time.")
print_time(time.time()-starttime)













