# Copyright (C) 2019 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, srk.py, is a python library that calculates                   #
# thermodynamic properties such as V, P, H, S, U, and G from the Soave,    #
# Redlich, Kwong equation of state.                                        #
#                                                                          #
# srk.py is free software: you can redistribute it and/or                  #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# srk.py is distributed in the hope that it will be useful,                #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with thermoproperties.py.  If not, see                             #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# srk.py                                                                   #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - October 2019                                               #
# Version 2.0 - May 2021 Changed name form srkprops to srk                 #
# ======================================================================== #
"""
This module contains functions that calculate thermodynamic properties  
at a given T and P using the Soave-Redlich-Kwong equation of state.     
Functions are also available that calculate various partial derivatives of  
P as a function of T and v. The functions require the critical             
temperature, pressure, and acentric factor of the compound of interest.  
                                                                         
The equations follow that presented in Chapter 4 of The Properties of    
Gases and Liquids, 5th ed. by Poling, Prausnitz, and O'Connell.  The      
volume is solved by placing the equations on state in the dimensionless  
cubic form.  For more information, see [1].

This module is part of the dippr package.              

Import the module using

  import dippr.eos.srk  

Imported in the manner, the functions can be called as  
  
  srk.vl(t,p,tc,pc,w)

which would return the liquid volume at t and p for the compound described
by tc, pc, and w.

Many functions are available, but several as auxillary function that 
are used in the main functions. Below, the lists of input parameters
is found followed by the list of functions that are most useful.   

----------------------------------------------------------------------
DEFINITION OF INPUT PARAMETERS FOR FUNCTIONS                          
----------------------------------------------------------------------
Symbol              Property                                Units       
----------------------------------------------------------------------
t                   system temperature                      K           
p                   system pressure                         Pa          
v                   system molar volume                     m**3/mol    
tc                  critical temperature of compound        K           
pc                  critical pressure of compound           Pa          
w                   acentric factor of compound             unitless    
----------------------------------------------------------------------
                                                                      
                                                                      
----------------------------------------------------------------------
PROPERTY FUNCTIONS                                          
----------------------------------------------------------------------
Function            Return Value                            Units       
----------------------------------------------------------------------
vl(t,p,tc,pc,w)     liquid molar volume                     m**3/mol    
vv(t,p,tc,pc,w)     vapor molar volume                      m**3/mol    
zl(t,p,tc,pc,w)     liquid compressibility                  unitless    
zv(t,p,tc,pc,w)     vapor compressibility                   unitless
hrl(t,p,tc,pc,w)    liquid residual enthalpy                J/mol 
hrv(t,p,tc,pc,w)    vapor residual enthalpy                 J/mol   
srl(t,p,tc,pc,w)    liquid residual entropy                 J/mol/K 
srv(t,p,tc,pc,w)    vapor residual entropy                  J/mol/K 
arl(t,p,tc,pc,w)    liquid residual Helmholtz energy        J/mol 
arv(t,p,tc,pc,w)    vapor residual Helmholtz energy         J/mol 
grl(t,p,tc,pc,w)    liquid residual Gibbs energy            J/mol 
grv(t,p,tc,pc,w)    vapor residual Gibbs energy             J/mol 
lnphil(t,p,tc,pc,w) natural log liquid fugacity coefficient unitless
lnphiv(t,p,tc,pc,w) natural log vapor fugacity coefficient  unitless
                                                                     
The functions listed above have been validated against the data in    
the tables on pages 6.11 and 6.12 of the 5th edition of The Properties
of Gases and Liquids [1].                                                 

References
----------
.. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
   of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).
"""
import numpy as np

# gas constant in J/mol/K
rg = 8.31447215 

# ------------------------------------------------------------------------ #
# Auxillary functions to calculate the quantities to place the EOS into    #
# the cubic z form.                                                        #
# ------------------------------------------------------------------------ #

def kappa(w):
	r""" kappa parameter for the SRK EOS
	
	Parameters
	----------
	w : float
		acentric factor of the compound (unitless)

	Returns
	-------
	float
		kappa = 0.480 + 1.574*w - 0.176*w**2 (unitless) 	
	
	""" 
	x = 0.480 + 1.574*w - 0.176*w**2
	return(x)

def alpha(t,tc,w):
	r""" alpha function for the SRK EOS
	
	Parameters
	----------
	t : float
		system temperature (K)
	
	tc : float
		critical temperature of the compound (K)
	
	w : float
		acentric factor of the compound (unitless)

	Returns
	-------
	float
		alpha = (1.0+kappa(w)*(1.0-(t/tc)**0.5))**2 (unitless)	
	
	""" 
	x = (1.0+kappa(w)*(1.0-(t/tc)**0.5))**2
	return(x)

def a(tc,pc):
	r""" a parameter for the SRK EOS
	
	Parameters
	----------
	tc : float
		critical temperature of the compound (K)
	
	pc : float
		critical pressure of the compound (Pa)

	Returns
	-------
	float
		a = 0.42748*rg**2*tc**2/pc (Pa*m**6/mol**2)	
	
	""" 
	x = 0.42748*rg**2*tc**2/pc
	return(x)

def b(tc,pc):
	r""" b parameter for the SRK EOS
	
	Parameters
	----------
	tc : float
		critical temperature of the compound (K)
	
	pc : float
		critical pressure of the compound (Pa)

	Returns
	-------
	float
		b = 0.08664*rg*tc/pc (m**3/mol)	
	
	""" 
	x = 0.08664*rg*tc/pc
	return(x)

def ThetaPrime(t,p,tc,pc,w): # Theta prime for the cubic form of the SRK EOS
    x = a(tc,pc)*alpha(t,tc,w)*p/(rg*t)**2
    return(x) # dimenionless

def dThetadT(t,tc,pc,w): # the temperature derivative of ThetaPrime
    x = -1.0*a(tc,pc)*kappa(w)*np.sqrt(alpha(t,tc,w)/t/tc)
    return(x) # Pa*m**6/mol**2/K

def d2ThetadT2(t,tc,pc,w): # the second temperature derivative of ThetaPrime
	sqrttc = np.sqrt(tc)
	x = 0.5*a(tc,pc)*kappa(w)/sqrttc/t*(np.sqrt(alpha(t,tc,w)/t)+kappa(w)/sqrttc)
	return(x) # Pa*m**6/mol**2/K**2

def BPrime(t,p,tc,pc): # B prime for the cubic form of the SRK EOS
    x = b(tc,pc)*p/rg/t
    return(x) # dimensionless

def deltaPrime(t,p,tc,pc): # delta prime for the cubic form of the SRK EOS
    x = b(tc,pc)*p/rg/t
    return(x) # dimensionless    

def epsilonPrime(t,p,tc,pc): # epsilon prime for the cubic form of the SRK EOS
	x = 0.0 * (p/rg/t)**2
	return(x) # dimensionless

def etaPrime(t,p,tc,pc): # eta prime for the cubic form of the SRK EOS
    x = b(tc,pc)*p/rg/t
    return(x) # dimensionless

# ------------------------------------------------------------------------ #
# Compressibility Functions                                                #
# ------------------------------------------------------------------------ # 
def zl(t,p,tc,pc,w): # liquid compressibility at t and p
    y = np.zeros(4)
    y[0] = 1.0
    y[1] = deltaPrime(t,p,tc,pc) - BPrime(t,p,tc,pc) - 1.0
    y[2] = ThetaPrime(t,p,tc,pc,w) + epsilonPrime(t,p,tc,pc)-deltaPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0)
    y[3] = -1.0*(epsilonPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0) + ThetaPrime(t,p,tc,pc,w)*etaPrime(t,p,tc,pc))
    r = np.roots(y)
    for i in range(3):
        if(np.imag(r[i]) != 0.0): r[i] = 10**308
    x = np.real(np.sort(r))
    return(x[0]) # dimenionless

def zv(t,p,tc,pc,w): # vapor compressibility at t and p
    y = np.zeros(4)
    y[0] = 1.0
    y[1] = deltaPrime(t,p,tc,pc) - BPrime(t,p,tc,pc) - 1.0
    y[2] = ThetaPrime(t,p,tc,pc,w) + epsilonPrime(t,p,tc,pc)-deltaPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0)
    y[3] = -1.0*(epsilonPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0) + ThetaPrime(t,p,tc,pc,w)*etaPrime(t,p,tc,pc))
    r = np.roots(y)
    for i in range(3):
        if(np.imag(r[i]) != 0.0): r[i] = 0.0
    x = np.real(np.sort(r))
    return(x[2]) # dimensionless 

# ------------------------------------------------------------------------ #
# The Pressure Function                                                    #
# ------------------------------------------------------------------------ #
def P(t,v,tc,pc,w): # The pressure at t and v from the SRK EOS
	bb = b(tc,pc)
	x = rg*t/(v-bb) - a(tc,pc)*alpha(t,tc,w)/(v*(v+bb))
	return(x) # Pa

# ------------------------------------------------------------------------ #
# Molar Volume Functions                                                   #
# ------------------------------------------------------------------------ #
def vl(t,p,tc,pc,w):  # liquid molar volume at t and p
    x = zl(t,p,tc,pc,w)*rg*t/p
    return(x) # m**3/mol

def vv(t,p,tc,pc,w):  # vapor molar volume at t and p
    x = zv(t,p,tc,pc,w)*rg*t/p
    return(x) # m**3/mol
	
# ------------------------------------------------------------------------ #
# Partial Derivative Functions                                             #
# ------------------------------------------------------------------------ #               
def dPdV(t,v,tc,pc,w): # the partial derivative of P with respect to v at constant T
	aalpha = a(tc,pc)*alpha(t,tc,w)
	bb = b(tc,pc)
	x = aalpha*(2*v+bb)/(v**2*(v+bb)**2) - rg*t/(v-bb)**2
	return(x) # Pa*mol/m**3
	
def dPdT(t,v,tc,pc,w): # the partial derivative of P with respect to T at constant v
	bb = b(tc,pc)
	x = rg/(v-bb) - dThetadT(t,tc,pc,w)/(v*(v+bb))
	return(x) # Pa/K 
	
def dVdT(t,v,tc,pc,w): # the partial derivative of v with respect to T at constant P
	return(-1.0*dPdT(t,v,tc,pc,w)/dPdV(t,v,tc,pc,w)) # m**3/mol/K

# ------------------------------------------------------------------------ #
# Residual Property Functions                                              #
# ------------------------------------------------------------------------ #

# the functions to obtain the liquid and vapor residual enthalpy
def hrl(t,p,tc,pc,w): # liquid residual enthalpy at t and p from SRK EOS
    x =(t*dThetadT(t,tc,pc,w) - a(tc,pc)*alpha(t,tc,w))/b(tc,pc)*np.log(1 + BPrime(t,p,tc,pc)/zl(t,p,tc,pc,w)) + rg*t*(zl(t,p,tc,pc,w)-1.0)
    return(x) # J/mol

def hrv(t,p,tc,pc,w): # liquid residual enthalpy at t and p from SRK EOS
    x =(t*dThetadT(t,tc,pc,w) - a(tc,pc)*alpha(t,tc,w)) / b(tc,pc)*np.log(1 + BPrime(t,p,tc,pc)/zv(t,p,tc,pc,w)) + rg*t*(zv(t,p,tc,pc,w)-1.0)
    return(x) # J/mol

# the functions to obtain the liquid and vapor residual entropy
def srl(t,p,tc,pc,w): # liquid residual entropy at t and p from SRK EOS
    x = dThetadT(t,tc,pc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zl(t,p,tc,pc,w)) + rg*np.log(zl(t,p,tc,pc,w)-BPrime(t,p,tc,pc))
    return(x) # J/mol/K

def srv(t,p,tc,pc,w): # liquid residual entropy at t and p from SRK EOS
    x = dThetadT(t,tc,pc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zv(t,p,tc,pc,w)) + rg*np.log(zv(t,p,tc,pc,w)-BPrime(t,p,tc,pc))
    return(x) # J/mol/K

# the functions to obtain the liquid and vapor residual Helmholtz energy
def arl(t,p,tc,pc,w): # liquid residual Helmholtz energy at t and p from SRK EOS
    x = -1.0*a(tc,pc)*alpha(t,tc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zl(t,p,tc,pc,w)) - rg*t*np.log(zl(t,p,tc,pc,w)-BPrime(t,p,tc,pc))
    return(x) # J/mol

def arv(t,p,tc,pc,w): # liquid residual Helmholtz energy at t and p from SRK EOS
    x = -1.0*a(tc,pc)*alpha(t,tc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zv(t,p,tc,pc,w)) - rg*t*np.log(zv(t,p,tc,pc,w)-BPrime(t,p,tc,pc))
    return(x) # J/mol

# the functions to obtain the liquid and vapor fugacity coefficient
def lnphil(t,p,tc,pc,w): # liquid fugacity coefficient at t and p from SRK EOS
    x = -1.0/rg/t*a(tc,pc)*alpha(t,tc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zl(t,p,tc,pc,w)) - np.log(zl(t,p,tc,pc,w)-BPrime(t,p,tc,pc)) + zl(t,p,tc,pc,w) - 1.0
    return(x) # dimensionless

def lnphiv(t,p,tc,pc,w): # liquid fugacity coefficient at t and p from SRK EOS
    x = -1.0/rg/t*a(tc,pc)*alpha(t,tc,w)/b(tc,pc)*np.log(1.0 + BPrime(t,p,tc,pc)/zv(t,p,tc,pc,w)) - np.log(zv(t,p,tc,pc,w)-BPrime(t,p,tc,pc)) + zv(t,p,tc,pc,w) - 1.0
    return(x) # dimensionless

# the functions to obtain the liquid and vapor residual Gibbs energy
def grl(t,p,tc,pc,w): # liquid residual gibbs energy at t and p from SRK EOS
    x = rg*t*lnphil(t,p,tc,pc,w)
    return(x) # J/mol

def grv(t,p,tc,pc,w): # vapor residual gibbs energy at t and p from SRK EOS
    x = rg*t*lnphiv(t,p,tc,pc,w)
    return(x) # J/mol
	

