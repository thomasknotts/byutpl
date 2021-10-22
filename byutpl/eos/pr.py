# Copyright (C) 2021 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, pr.py, is a python library that calculates                    #
# thermodynamic properties such as V, P, H, S, U, and G from the Peng      #
# Robinson equation of state.                                              #
#                                                                          #
# pr.py is free software: you can redistribute it and/or                   #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# pr.py is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with thermoproperties.py.  If not, see                             #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# pr.py                                                                    #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - June 2021                                                  #
# ======================================================================== #
"""
This module contains functions that calculate thermodynamic properties  
at a given T and P using the Peng-Robinson equation of state. [1]     
Functions are also available that calculate various partial derivatives of  
P as a function of T and V. The functions require the critical             
temperature, pressure, and acentric factor of the compound of interest.  

The equations follow that presented in Chapter 4 of The Properties of    
Gases and Liquids, 5th ed. by Poling, Prausnitz, and O'Connell.  The      
volume (compressibility) is solved by placing the equations on state in the
dimensionless cubic z form.

    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0

For more information, see [2].

This module is part of the byutpl package.              

Import the module using

  import byutpl.eos.pr as pr

Loaded in this manner, the functions can be called with syntax like the 
following.  
  
  pr.vl(t,p,tc,pc,w)

This would return the liquid molar volume from the PR EOS at `t` and `p`
for the compound described by `tc`, `pc`, and `w`.

Many functions are available, but several are support functions that 
are used by the main functions and are rarely called by the user.
Below, the lists of input parameters is found followed by the list of 
functions that are most useful. Use help(pr) for a list of all functions
available in the module.    

======================================================================
INPUT PARAMETERS FOR FUNCTIONS                          
======================================================================
Symbol              Property                                Units       
----------------------------------------------------------------------
t                   system temperature                      K           
p                   system pressure                         Pa          
v                   system molar volume                     m**3/mol    
tc                  critical temperature of compound        K           
pc                  critical pressure of compound           Pa          
w                   acentric factor of compound             unitless    
======================================================================


======================================================================
PROPERTY FUNCTIONS                                          
======================================================================
Function            Return Value                            Units       
----------------------------------------------------------------------
vl(t,p,tc,pc,w)     liquid molar volume                     m**3/mol    
vv(t,p,tc,pc,w)     vapor molar volume                      m**3/mol    
zl(t,p,tc,pc,w)     liquid compressibility                  unitless    
zv(t,p,tc,pc,w)     vapor compressibility                   unitless
hrl(t,p,tc,pc,w)    liquid residual enthalpy                J mol**-1 K**-1 
hrv(t,p,tc,pc,w)    vapor residual enthalpy                 J mol**-1 K**-1   
srl(t,p,tc,pc,w)    liquid residual entropy                 J mol**-1 K**-1 
srv(t,p,tc,pc,w)    vapor residual entropy                  J mol**-1 K**-1 
arl(t,p,tc,pc,w)    liquid residual Helmholtz energy        J/mol 
arv(t,p,tc,pc,w)    vapor residual Helmholtz energy         J/mol 
grl(t,p,tc,pc,w)    liquid residual Gibbs energy            J/mol 
grv(t,p,tc,pc,w)    vapor residual Gibbs energy             J/mol 
lnphil(t,p,tc,pc,w) natural log liquid fugacity coefficient unitless
lnphiv(t,p,tc,pc,w) natural log vapor fugacity coefficient  unitless
cprl(t,p,tc,pc,w)   liquid residual isobaric heat capacity  J mol**-1 K**-1
cprv(t,p,tc,pc,w)   vapor residual isobaric heat capacity   J mol**-1 K**-1
======================================================================

The residual for property J is defined as
  J - J*
where J is the value for the real fluid and J* is the value for the
ideal gas.                                              

References
----------
.. [1] D-Y Peng and D. B. Robinson, A New Two-Constant Equation of State,
   Ind. Eng. Chem. Fundam., 15(1) 59-64 (1976).

.. [2] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
   of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).

"""
import numpy as np

# gas constant in J mol**-1 K**-1

rg = 8.31447215 

# ------------------------------------------------------------------------ #
# Parameters for the equation of state.                                    #
# ------------------------------------------------------------------------ #

def kappa(w):
    """ kappa parameter for the PR EOS (unitless)

    Parameters
    ----------
    w : float
        acentric factor of the compound (unitless)

    Returns
    -------
    float
        kappa = 0.37464 + 1.54226*w - 0.26992*w**2    (unitless)     
    """ 
    x = 0.37464 + 1.54226*w - 0.26992*w**2
    return(x)

def alpha(t,tc,w):
    """ alpha function for the PR EOS (unitless)
    
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
        alpha = (1.0+kappa*(1.0-Tr**0.5))**2    (unitless)    
    
    """ 
    x = (1.0+kappa(w)*(1.0-(t/tc)**0.5))**2
    return(x)

def a(tc,pc):
    """ a parameter for the PR EOS in units of Pa*m**6/mol**2
    
    Parameters
    ----------
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)

    Returns
    -------
    float
        a = 0.45724*rg**2*tc**2/pc    (Pa*m**6/mol**2)    
    """ 
    x = 0.45724*rg**2*tc**2/pc
    return(x)

def b(tc,pc):
    """ b parameter for the PR EOS in units of m**3/mol
    
    Parameters
    ----------
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)

    Returns
    -------
    float
        b = 0.07780*rg*tc/pc    (m**3/mol)        
    """ 
    x = 0.07780*rg*tc/pc
    return(x)

# ------------------------------------------------------------------------ #
# Supporting functions to place the equation of state in the cubic z form. #
# ------------------------------------------------------------------------ #

def ThetaPrime(t,p,tc,pc,w):
    """ ThetaPrime parameter for the PR EOS in cubic z form (unitless)

    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Determining z can be done by solving this third-order polynomial. This
    function returns ThetaPrime for the PR EOS which is needed for two of
    the coefficients.
    
    This function will usually not be called directly by a user.
    
    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)

    Returns
    -------
    float
        ThetaPrime = a*alpha*p/(rg*t)**2    (unitless)    
    
    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).
    """ 
    x = a(tc,pc)*alpha(t,tc,w)*p/(rg*t)**2
    return(x)

def dThetadT(t,tc,pc,w):
    """ first temperature derivative of the Theta parameter for the
    PR EOS in units of Pa*m**6/(mol**2*K)
    
    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
       
    Temperature derivatives of a cubic equation of state are needed for some 
    thermodynamic properties. This function returns the first temperature 
    derivative of TheataPrime. 
    
    This function will usually not be called directly by a user.
    
    Parameters
    ----------
    t : float
        system temperature (K)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)

    Returns
    -------
    float
        dThetadT = -a*kappa*(alpha/(t*tc))**0.5    (Pa*m**6/(mol**2*K))

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).
    """ 
    x = -1.0*a(tc,pc)*kappa(w)*np.sqrt(alpha(t,tc,w)/t/tc)
    return(x)

def d2ThetadT2(t,tc,pc,w):
    """ second temperature derivative of the Theta parameter for the
    PR EOS in units of Pa*m**6/(mol**2*K**2)
    
    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Temperature derivatives of a cubic equation of state are needed for some 
    thermodynamic properties. This function returns the second temperature 
    derivative of TheataPrime. 
    
    This function will usually not be called directly by a user.
    
    Parameters
    ----------
    t : float
        system temperature (K)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)

    Returns
    -------
    float
        dThetadT = 0.5*a(tc,pc)*kappa(w)/sqrttc/t*(np.sqrt(alpha(t,tc,w)/t) +
        kappa(w)/sqrttc)    (Pa*m**6/(mol**2*K**2))

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).    
    """ 
    sqrttc = np.sqrt(tc)
    x = 0.5*a(tc,pc)*kappa(w)/sqrttc/t*(np.sqrt(alpha(t,tc,w)/t) + \
        kappa(w)/sqrttc)
    return(x)

def BPrime(t,p,tc,pc):
    """ BPrime parameter for the PR EOS in cubic z form (unitless)

    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Determining z can be done by solving this third-order polynomial. This
    function returns BPrime for the PR EOS which is needed for three of
    the coefficients.

    This function will usually not be called directly by a user.
        
    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    Returns
    -------
    float
        BPrime = b*p/(rg*t)    (unitless)

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).    
    """ 
    x = b(tc,pc)*p/rg/t
    return(x)

def deltaPrime(t,p,tc,pc):
    """ deltaPrime parameter for the PR EOS in cubic z form (unitless)

    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Determining z can be done by solving this third-order polynomial. This
    function returns deltaPrime for the PR EOS which is needed for two of
    the coefficients.
    
    This function will usually not be called directly by a user.
        
    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    Returns
    -------
    float
        deltaPrime = 2*b*p/(rg*t)    (unitless)

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).    
    """ 
    x = 2*b(tc,pc)*p/rg/t
    return(x)    

def epsilonPrime(t,p,tc,pc):
    """ epsilonPrime parameter for the PR EOS in cubic z form (unitless)

    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Determining z can be done by solving this third-order polynomial. This
    function returns epsilonPrime for the PR EOS which is needed for two of
    the coefficients.
    
    This function will usually not be called directly by a user.
        
    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    Returns
    -------
    float
        epsilonPrime = -b(tc,pc)**2 * (p/rg/t)**2    (unitless)

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).    
    """ 
    x = -b(tc,pc)**2 * (p/rg/t)**2
    return(x)

def etaPrime(t,p,tc,pc):
    """ etaPrime parameter for the PR EOS in cubic z form (unitless)

    All cubic equations of state can be placed into a generalized dimensionless
    form that is cubic in z (compressibility) as shown below [1].
    
    z**3 + 
    (deltaPrime - BPrime -1)*z**2 +
    [ThetaPrime + epsilonPrime - deltaPrime*(BPrime + 1)]*z -
    [epsilonPrime*(BPrime + 1) + ThetaPrime*etaPrime] = 0
    
    Determining z can be done by solving this third-order polynomial. This
    function returns etaPrime for the PR EOS which is needed for two of
    the coefficients.
    
    This function will usually not be called directly by a user.
        
    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    Returns
    -------
    float
        etaPrime = b*p/(rg*t)    (unitless)

    References
    ----------
    .. [1] B. E Poling, J. M. Prausnitz, J. P. O'Connell, The Properties
       of Gases and Liquids 5th edition,  McGraw-Hill, New York (2001).
    """ 
    x = b(tc,pc)*p/rg/t
    return(x)

# ------------------------------------------------------------------------ #
# Compressibility Functions                                                #
# ------------------------------------------------------------------------ # 
    
def zl(t,p,tc,pc,w):
    """liquid compressibility from the PR EOS (unitless)

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        compressibility of the liquid phase at `t` and `p` for the compound
        described by `tc`, `pc`, and `w`    (unitless)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.zv(t,p,tc,pc,w)) will be equal.
    """
    y = np.zeros(4)
    y[0] = 1.0
    y[1] = deltaPrime(t,p,tc,pc) - BPrime(t,p,tc,pc) - 1.0
    y[2] = ThetaPrime(t,p,tc,pc,w) + epsilonPrime(t,p,tc,pc) - \
           deltaPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0)
    y[3] = -1.0*(epsilonPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0) + \
           ThetaPrime(t,p,tc,pc,w)*etaPrime(t,p,tc,pc))
    r = np.roots(y)
    for i in range(3):
        if(np.imag(r[i]) != 0.0): r[i] = 10**308
    x = np.real(np.sort(r))
    return(x[0])

def zv(t,p,tc,pc,w):
    """vapor compressibility from the PR EOS (unitless)

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        compressibility of the vapor phase at `t` and `p` for the compound
        described by `tc`, `pc`, and `w`    (unitless)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.zl(t,p,tc,pc,w)) will be equal.  
    """
    y = np.zeros(4)
    y[0] = 1.0
    y[1] = deltaPrime(t,p,tc,pc) - BPrime(t,p,tc,pc) - 1.0
    y[2] = ThetaPrime(t,p,tc,pc,w) + epsilonPrime(t,p,tc,pc) - \
           deltaPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0)
    y[3] = -1.0*(epsilonPrime(t,p,tc,pc)*(BPrime(t,p,tc,pc) + 1.0) + \
           ThetaPrime(t,p,tc,pc,w)*etaPrime(t,p,tc,pc))
    r = np.roots(y)
    for i in range(3):
        if(np.imag(r[i]) != 0.0): r[i] = 0.0
    x = np.real(np.sort(r))
    return(x[2]) 

# ------------------------------------------------------------------------ #
# The Pressure Function                                                    #
# ------------------------------------------------------------------------ #
    
def P(t,v,tc,pc,w):
    """pressure from the PR EOS in units of Pa

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        pressure of the system at `t` and `v` for the compound
        described by `tc`, `pc`, and `w`    (Pa)       
    """
    bb = b(tc,pc)
    x = rg*t/(v-bb) - a(tc,pc)*alpha(t,tc,w)/(v**2 + 2*bb*v - bb**2)
    return(x)

# ------------------------------------------------------------------------ #
# Molar Volume Functions                                                   #
# ------------------------------------------------------------------------ #
    
def vl(t,p,tc,pc,w):
    """liquid molar volume from the PR EOS in units of m**3/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid molar volume of the system at `t` and `p` for the compound
        described by `tc`, `pc`, and `w`    (m**3/mol)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.vv(t,p,tc,pc,w)) will be equal.
    """
    x = zl(t,p,tc,pc,w)*rg*t/p
    return(x)

def vv(t,p,tc,pc,w):
    """vapor molar volume from the PR EOS in units of m**3/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor molar volume of the system at `t` and `p` for the compound
        described by `tc`, `pc`, and `w`    (m**3/mol)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.vl(t,p,tc,pc,w)) will be equal.
    """
    x = zv(t,p,tc,pc,w)*rg*t/p
    return(x)
    
# ------------------------------------------------------------------------ #
# Partial Derivative Functions                                             #
# ------------------------------------------------------------------------ #

def dPdV(t,v,tc,pc,w):
    """the partial derivative of pressure with respect to molar volume at 
    constant temperature for the PR EOS in units of Pa*mol/m**3

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        the partial derivative of the pressure with respect to `v` at
        constant `t` for the PR EOS for the compound described by 
        `tc`, `pc`, and `w`; the phase correspond to the phase of `v`
        (Pa*mol/m**3) 
    """
    aalpha = a(tc,pc)*alpha(t,tc,w)
    bb = b(tc,pc)
    x = aalpha*2.0*(v+bb)/(v**2+2.0*v*bb - bb**2)**2 - rg*t/(v-bb)**2
    return(x)
    
def dPdT(t,v,tc,pc,w):
    """the partial derivative of pressure with respect to temperature at 
    constant molar volume for the PR EOS in units of Pa/K

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        the partial derivative of the pressure with respect to `t` at
        constant `v` for the PR EOS for the compound described by 
        `tc`, `pc`, and `w`; the phase correspond to the phase of `v`
        (Pa/K) 
    """
    bb = b(tc,pc)
    x = rg/(v-bb) - dThetadT(t,tc,pc,w)/(v**2 + 2.0*v*bb - bb**2)
    return(x) 
    
def dVdT(t,v,tc,pc,w):
    """the partial derivative of molar volume with respect to temperature at 
    constant pressure for the PR EOS in units of m**3/(mol*K)

    Parameters
    ----------
    t : float
        system temperature (K)
        
    v : float
        system molar volume (m**3/mol)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        the partial derivative of the molar volume with respect to `t` at
        constant `p` for the PR EOS for the compound described by 
        `tc`, `pc`, and `w`; the phase correspond to the phase of `v`
        (m**3/(mol*K)) 
    """
    return(-1.0*dPdT(t,v,tc,pc,w)/dPdV(t,v,tc,pc,w))

# ------------------------------------------------------------------------ #
# Residual Property Functions                                              #
# ------------------------------------------------------------------------ #
    
def hrl(t,p,tc,pc,w): 
    """liquid residual enthalpy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual enthalpy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.hrv(t,p,tc,pc,w)) will be equal. 
    """
    sqrt2=np.sqrt(2.0)
    z = zl(t,p,tc,pc,w)
    x = (t*dThetadT(t,tc,pc,w) - a(tc,pc)*alpha(t,tc,w))/ \
        (2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) + \
        rg*t*(z-1.0)
    return(x)

def hrv(t,p,tc,pc,w):
    """vapor residual enthalpy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual enthalpy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.hrl(t,p,tc,pc,w)) will be equal. 
    """
    sqrt2=np.sqrt(2.0)
    z = zv(t,p,tc,pc,w)
    x = (t*dThetadT(t,tc,pc,w) - a(tc,pc)*alpha(t,tc,w))/ \
        (2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) + \
        rg*t*(z-1.0)
    return(x)


def srl(t,p,tc,pc,w):
    """liquid residual entropy from the PR EOS in units of J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual entropy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.srv(t,p,tc,pc,w)) will be equal. 
    """    
    sqrt2=np.sqrt(2.0)
    z = zl(t,p,tc,pc,w)
    x = dThetadT(t,tc,pc,w)/(2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) + \
        rg*np.log(z-BPrime(t,p,tc,pc))
    return(x)

def srv(t,p,tc,pc,w):
    """vapor residual entropy from the PR EOS in units of J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual entropy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.srl(t,p,tc,pc,w)) will be equal. 
    """
    sqrt2=np.sqrt(2.0)
    z = zv(t,p,tc,pc,w)
    x = dThetadT(t,tc,pc,w)/(2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) + \
        rg*np.log(z-BPrime(t,p,tc,pc))
    return(x)

def arl(t,p,tc,pc,w):
    """liquid residual Helmholtz energy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual Helmholtz energy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.arv(t,p,tc,pc,w)) will be equal. 
    """    
    h = hrl(t,p,tc,pc,w)
    s = srl(t,p,tc,pc,w)
    z = zl(t,p,tc,pc,w)
    x = h - rg*t*(z - 1.0) - t*s
    return(x)

def arv(t,p,tc,pc,w):
    """vapor residual Helmholtz energy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual Helmholtz energy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.arl(t,p,tc,pc,w)) will be equal. 
    """
    h = hrv(t,p,tc,pc,w)
    s = srv(t,p,tc,pc,w)
    z = zv(t,p,tc,pc,w)
    x = h - rg*t*(z - 1.0) - t*s
    return(x)


def lnphil(t,p,tc,pc,w):
    """natural logarithm of the liquid phase fugacity coefficient from the
    PR EOS (unitless)

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        natural logarithm of the liquid phase fugacity coefficient of the
        system at `t` and `p` for the compound described by `tc`, `pc`, 
        and `w`    (unitless)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.lnphiv(t,p,tc,pc,w)) will be equal. The two values will 
        also be equal if the system is in vapor/liquid equilibrium.
    """    
    sqrt2 = np.sqrt(2.0)
    z = zl(t,p,tc,pc,w)
    x = -1.0/rg/t*a(tc,pc)*alpha(t,tc,w)/(2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) - \
        np.log(z - BPrime(t,p,tc,pc)) + z - 1.0
    return(x)

def lnphiv(t,p,tc,pc,w):
    """natural logarithm of the vapor phase fugacity coefficient from the
    PR EOS (unitless)

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        natural logarithm of the vapor phase fugacity coefficient of the
        system at `t` and `p` for the compound described by `tc`, `pc`, 
        and `w`    (unitless)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.lnphil(t,p,tc,pc,w)) will be equal. The two values will 
        also be equal if the system is in vapor/liquid equilibrium.
    """
    sqrt2 = np.sqrt(2.0)
    z = zv(t,p,tc,pc,w)
    x = -1.0/rg/t*a(tc,pc)*alpha(t,tc,w)/(2.0*sqrt2*b(tc,pc))* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc))) - \
        np.log(z - BPrime(t,p,tc,pc)) + z - 1.0
    return(x)


def grl(t,p,tc,pc,w):
    """liquid residual Gibbs energy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual Gibbs energy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.grv(t,p,tc,pc,w)) will be equal. 
    """    
    x = rg*t*lnphil(t,p,tc,pc,w)
    return(x)

def grv(t,p,tc,pc,w):
    """vapor residual Gibbs energy from the PR EOS in units of J/mol

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual Gibbs energy of the system at `t` and `p` for the 
        compound described by `tc`, `pc`, and `w`    (J/mol)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.grl(t,p,tc,pc,w)) will be equal. 
    """    
    x = rg*t*lnphiv(t,p,tc,pc,w)
    return(x)

def cvrv(t,p,tc,pc,w):
    """vapor residual isochoric heat capacity from the PR EOS in units of
       J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual isochoric heat capcity of the system at `t`
        and `p` for the compound described by `tc`, `pc`, and `w`
        (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.cprl(t,p,tc,pc,w)) will be equal. 
    """
    z = zv(t,p,tc,pc,w)
    sqrt2 = np.sqrt(2.0)
    x = t/(2.0*sqrt2*b(tc,pc))*d2ThetadT2(t,tc,pc,w)* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc)))
    return(x)
    
def cvrl(t,p,tc,pc,w):
    """liquid residual isochoric heat capacity from the PR EOS in units of
       J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual isochoric heat capcity of the system at `t`
        and `p` for the compound described by `tc`, `pc`, and `w`
        (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.cprv(t,p,tc,pc,w)) will be equal. 
    """
    z = zl(t,p,tc,pc,w)
    sqrt2 = np.sqrt(2.0)
    x = t/(2.0*sqrt2*b(tc,pc))*d2ThetadT2(t,tc,pc,w)* \
        np.log((z + (1.0+sqrt2)*BPrime(t,p,tc,pc))/ \
        (z + (1.0-sqrt2)*BPrime(t,p,tc,pc)))
    return(x)
    
def cprv(t,p,tc,pc,w):
    """vapor residual isobaric heat capacity from the PR EOS in units of
       J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        vapor residual isobaric heat capcity of the system at `t`
        and `p` for the compound described by `tc`, `pc`, and `w`
        (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the vapor
        value (from this function) and the liquid value 
        (from pr.cprl(t,p,tc,pc,w)) will be equal. 
    """
    v = vv(t,p,tc,pc,w)
    bb = b(tc,pc)
    x = cvrv(t,p,tc,pc,w) - \
        t*(rg/(v - bb) - dThetadT(t,tc,pc,w)/(v**2 + 2*v*bb - bb**2))**2/ \
        (-rg*t/(v - bb)**2 + \
        (a(tc,pc)*alpha(t,tc,w)*2.0*(v + bb))/(v**2 + 2*v*bb - bb**2)**2) - \
        rg
    return(x)
    
def cprl(t,p,tc,pc,w):
    """liquid residual isobaric heat capacity from the PR EOS in units of
       J mol**-1 K**-1

    Parameters
    ----------
    t : float
        system temperature (K)
        
    p : float
        system pressure (Pa)
        
    tc : float
        critical temperature of the compound (K)
    
    pc : float
        critical pressure of the compound (Pa)
        
    w : float
        acentric factor of the compound (unitless)
        
    Returns
    -------
    float
        liquid residual isobaric heat capcity of the system at `t`
        and `p` for the compound described by `tc`, `pc`, and `w`
        (J mol**-1 K**-1)
        
        If only one phase exists for the input conditions, both the liquid
        value (from this function) and the vapor value 
        (from pr.cprv(t,p,tc,pc,w)) will be equal. 
    """
    v = vl(t,p,tc,pc,w)
    bb = b(tc,pc)
    x = cvrl(t,p,tc,pc,w) - \
        t*(rg/(v - bb) - dThetadT(t,tc,pc,w)/(v**2 + 2*v*bb - bb**2))**2/ \
        (-rg*t/(v - bb)**2 + \
        (a(tc,pc)*alpha(t,tc,w)*2.0*(v + bb))/(v**2 + 2*v*bb - bb**2)**2) - \
        rg
    return(x)
