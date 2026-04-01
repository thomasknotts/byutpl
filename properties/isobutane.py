# Copyright (C) 2026 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, isobutane.py, is a python module of the                       #
# thermophysical properties of isobutane.  The properties, both            #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# isobutane.py is distributed in the hope that it will be useful,          #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with isobutane.py.  If not, see                                    #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this module, or other property data    #
# from the DIPPR(R) database, should include the citation below.           #
# W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley,                  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2025).                     #
#                                                                          #
# ======================================================================== #
# isobutane.py                                                             #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - January 2026                                               #
# ======================================================================== #
"""
This library contains functions for the properties of isobutane.
The values come from the DIPPR(R) Sample database [1], 
and the DIPPR(R) abbreviations are used. Vapor properties that are 
dependent on pressure are obtained using the Soave-Redlich-Kwong equation
of state.

This module is part of the byutpl package. Import the module using

  import byutpl.properties.isobutane as pen

When imported in this way, constant properties can be called as   

  pen.acen
  
which returns the acentric factor. Temperature dependent properties
can be called as
  
  pen.vtc(t)

which returns the vapor thermal conductivity at `t` where `t` is 
temperature in units of K. Temperature and pressure dependent properties
can be called as
  
  pen.vcp(t,p)

which returns the vapor heat capacity at `t` and `p` where 
`t` is temperature in units of K and `p` is pressure in units of Pa.
    
A complete list of properties, and the associated units, are found       
below.                                                                   

Function    Return Value                             Input Value(s)         
---------   --------------------------------------   ----------------- 
tc          critical temperature in K                none              
pc          critical pressure in Pa                  none              
vc          critical volume in m**3/mol              none              
zc          critical compress. factor (unitless)     none              
mw          molecular weight in kg/mol               none              
acen        acentric factor (unitless)               none              
ldn(t)      liquid density in kg/m**3                temperature in K  
icp(t)      ideal gas heat capacity in J/(mol*K)     temperature in K  
lcp(t)      liquid heat capacity in J/(mol*K)        temperature in K  
ltc(t)      liquid thermal conductivity in W/(m*K)   temperature in K  
vp(t)       liquid vapor pressure in Pa              temperature in K  
hvp(t)      heat of vaporization in J/mol            temperature in K  
lpr(t)      liquid Prandtl number (unitless)         temperature in K  
lvs(t)      liquid viscosity in Pa*s                 temperature in K  
lnu(t)      liquid kinematic viscosity in m**2/s     temperature in K  
tsat(p)     temperature at saturation in K           pressure in Pa    
vvs(t)      vapor (steam) viscosity in Pa*s          temperature in K  
vtc(t)      vapor (steam) therm. conduct. in W/(m*K) temperature in K  
vdn(t,p)    vapor (steam) density in kg/m**3         temperature in K
                                                     pressure in Pa
vcp(t,p)    vapor (steam) isobaric heat capacity     temperature in K
            in J/(mol*K)                             pressure in Pa
vnu(t,p)    vapor (steam) kinematic viscosity        temperature in K
            in m**2/s                                pressure in Pa
vpr(t,p)    vapor (steam) Prandtl number (unitless)  temperature in K
                                                     pressure in Pa

References
----------
.. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, 
   DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
   for Physical Properties, AIChE, New York, NY (2025).
"""

import numpy as np
from scipy.optimize import fsolve
import byutpl.eos.srk as srk
import byutpl.equations.dippreqns as dippr

# critical temperature
tc = 407.8 # units of K

# critical pressure
pc = 3.64e6 # units of Pa

# critical volume
vc = 0.000259 # units of m**3/mol

# critical compressibility factor
zc = 0.278 # unitless

# acentric factor
acen = 0.183521 # unitless

# molecular weight
mw = 58.1222e-3 # units of kg/mol
  
def ldn(t):
    """liquid density of isobutane 
	
    Liquid density of isobutane from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 105; valid from 113.54 - 407.8 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density
        of isobutane.

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of isobutane at `t`.
	"""
    c = np.array([1.0631,0.27506,407.8, 0.2758])
    y = dippr.eq105(t,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """liquid heat capacity of isobutane 
	
    Liquid heat capacity of isobutane from the DIPPR(R) correlation
    (Correlation B: DIPPR Equation 100; valid from 113.54 - 380.00 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity
        of isobutane.

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of isobutane at
        `t`.
	"""
    c = np.array([172370,-1783.9,14.759,-0.047909,5.805E-05])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """liquid thermal conductivity of isobutane 
	
    Liquid thermal conductivity of isobutane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 123; valid from 113.54 - 375 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal
        conductivity of isobutane.

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of isobutane at `t`.
	"""
    c = np.array([0.228978,-3.88363,6.36273,-2.68714])
    tau=1-t/tc
    y = dippr.eq123(tau,c)
    return(y)

def vp(t):
    """liquid vapor pressure of isobutane 
	
    Liquid vapor pressure of isobutane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 113.54 — 407.i8 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of
        isobutane.

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of isobutane at `t`.
	"""
    c = np.array([108.43,-5039.9,-15.012,0.022725,1])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t):
    """heat of vaporization of isobutane 
	
    Heat of vaporization of isobutane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 106; valid from 113.54 — 407.8 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat of vaporization of
        isobutane.

    Returns
    -------
    float
        The heat of vaporization (J/mol) of isobutane at `t`.
	"""
    c = np.array([39654000,1.274,-1.4255,0.60708,0])
    tr = t/tc
    y = dippr.eq106(tr,c)
    y = y / 1000 # convert from J/kmol to J/mol
    return(y)
    
def lvs(t):
    """liquid viscosity of isobutane 
	
    Liquid viscosity of isobutane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 110 - 310.95 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid viscosity of
        isobutane.

    Returns
    -------
    float
        The liquid viscosity (Pa*s) of isobutane at `t`.
	"""
    c = np.array([-13.912,797.09,0.45308,0,0])
    y = dippr.eq101(t,c)
    return(y)

def lnu(t):
    """liquid kinematic viscosity of isobutane 
	
    Liquid kinematic viscosity of isobutane calculated from the lvs and ldn
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid kinematic 
        viscosity of isobutane.

    Returns
    -------
    float
        The liquid kinematic viscosity (m**2/s) of isobutane at `t`.
	"""
    return(lvs(t)/ldn(t))

def lpr(t):
    """Prandtl number of liquid isobutane 
	
    Prandtl number of liquid isobutane calculated from the lcp, lvs, and ltc
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of
        liquid isobutane.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of liquid isobutane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return lcp(t)*lvs(t)/ltc(t)/mw

def ftsat(t,p):
    """function supplied to fsolve in tsat function 
	
    Function supplied to fsolve (in the f(x)=0 form) to solve for the 
    temperature at saturation for a given pressure.  This 
    function is of little use to users.  Users should use 
    the function tsat.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the function.  It is
    	the "x" value for which fsolve solves.
	
    p : float
    	The pressure (Pa) at which the saturated temperature is desired.

    Returns
    -------
    float
        The value of the function supplied to fsolve.  This value will be 
    	zero if 't' is the saturated temperature for `p`. 
	"""
    return(vp(t) - p)

def tsat(p):
    """saturated temperature for isobutane
	
    Saturation temperature of isobutane for a given pressure 'p'.  It is
    the temperature for which the following equation is true:
    vp(t)= `p`
    where vp is the function in this module and t is the value
    this function (tsat) returns.
	
    Parameters
    ----------
    p : float
        The pressure (Pa) at which to find the saturated temperature.

    Returns
    -------
    float
        The temperature (K) of isobutane at saturation at pressure `p`.
	"""
    x = 700 # guess in K
    y = fsolve(ftsat,x,p)
    return(y[0])
    
def vvs(t):
    """vapor viscosity of isobutane
	
    Vapor viscosity of isobutane at temperature `t` from the DIPPR(R)
    correlation.
    (Correlation A: DIPPR Equation 102; valid from 150 — 1000 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor viscosity of
        isobutane.

    Returns
    -------
    float
        The vapor viscosity of isobutane at `t`.
	"""
    c = np.array([1.0871E-07,0.78135,70.639,0.0])
    y = dippr.eq102(t,c)
    return(y)

def vtc(t):
    """vapor thermal conductivity of isobutane
	
    The vapor thermal conductivity of isobutane at temperature `t`
    from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 102; valid from 261.43 — 1000 K;
    uncertainty: < 10%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor thermal
        conductivity of isobutane.

    Returns
    -------
    float
        The vapor thermal conductivity of isobutane (W m**-1 K**-1) 
        at `t`.
	"""
    c = np.array([0.089772,0.18501,639.23,1114700])
    y = dippr.eq102(t,c)
    return(y)

def vdn(t,p):
    """vapor density of isobutane
	
    The vapor density of isobutane at temperature `t` and
    pressure `p` from the Soave-Redlich-Kwong equation of state. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor density of
        isobutane.

    p : float
        The pressure (Pa) at which to evaluate the vapor density of
        isobutane.

    Returns
    -------
    float
        The vapor density of isobutane (kg/m**3) at `t` and `p`.
	"""   
    v = srk.vv(t,p,tc,pc,acen)
    v = v / mw # convert from m**3/mol to m**3/kg
    return(1/v)
    
def icp(t): 
    """ideal gas heat capacity of isobutane
    
    The ideal gas heat capacity of isobutane at temperature `t` from the 
    DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 124; valid from 20 — 1500 K;
    uncertainty: < 1%)
    
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat
        capacity of isobutane.

    Returns
    -------
    float
        The ideal gas heat capacity of isobutane (J/(mol*K)) at `t`.
    """     
    c=np.array([33257.8886,48701.33848,411.25771,141642.72064,1544.73381,93408.54987,3850.61681])
    y=dippr.eq127(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def vcp(t,p):
    """vapor heat capacity isobutane
	
    Heat capacity of vapor isobutane calculated from the DIPPR(R) 
    correlation for ideal gas heat capacity and the residual property
    from the Soave-Redlich-Kwong equation of state.
    (ICP Correlation A: DIPPR Equation 127; valid from 20 — 1500 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat 
        capacity of vapor isobutane.

    p : float
        The pressure (Pa) at which to evaluate the heat capacity
        of vapor isobutane.

    Returns
    -------
    float
        The heat capacity (J/(mol*K)) of vapor isobutane
        at `t` and `p`.
	"""
    x = icp(t) + srk.cprv(t,p,tc,pc,acen)
    return(x)

def vnu(t,p):
    """vapor kinematic viscosity of isobutane
	
    Kinematic viscosity of vapor isobutane calculated from the vvs and
    vdn functions in this module. The calculation uses the Soave-Redlich-
    Kwong equation of state for the vapor density.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic 
        viscosity of vapor isobutane.

    p : float
        The pressure (Pa) at which to evaluate the kinematic viscosity
        of vapor isobutane.

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of vapor isobutane
        at `t` and `p`.
	"""
    return(vvs(t)/vdn(t,p))

def vpr(t, p):
    """Prandtl number of vapor isobutane
	
    Prandtl number of vapor isobutane calculated from the vcp, vvs, 
    and vtc functions in this module. The calculation uses the Soave-
    Redlich-Kwong equation of state to correct the ideal gas heat capacity
    to the real gas at `t` and `p`. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of vapor
        isobutane.

    p : float
        The pressure (Pa) at which to evaluate the Prandtl number of vapor
        isobutane.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of vapor isobutane at `t`
        and `p`.
	"""
    return(vcp(t,p)*vvs(t)/vtc(t)/mw)


def unit(key):
    """Returns the units of `key` 
	
    Returns the units of the constant or function in this module identified
    by 'key'
	
    Parameters
    ----------
    key : string
        The name of the constant or function in this module for which the
        units are needed.

    Returns
    -------
    string
        The units for the constant or function identified by `key`.
	"""
    if type(key) !=str:
        return('The parameter must be a string.')
    if key == 'tc':
        return('K')
    if key == 'pc':
        return('Pa')
    if key == 'vc':
        return('m**3/mol')
    if key == 'zc':
        return('unitless')
    if key == 'acen':
        return('unitless')
    if key == 'mw':
        return('kg/mol')
    if key == 'ldn':
        return('kg/m**3')
    if key == 'icp':
        return('J mol**-1 K**-1')
    if key == 'lcp':
        return('J mol**-1 K**-1')
    if key == 'vcp':
        return('J mol**-1 K**-1')
    if key == 'ltc':
        return('W m**-1 K**-1')
    if key == 'vp':
        return('Pa')
    if key == 'hvp':
        return('J/mol')
    if key == 'lpr':
        return('unitless')
    if key == 'lvs':
        return('Pa*s')
    if key == 'lnu':
        return('m**2/s')
    if key == 'tsat':
        return('K')
    if key == 'vvs':
        return('Pa*s')
    if key == 'vtc':
        return('W m**-1 K**-1')
    if key == 'vnu':
        return('m**2/s')
    if key == 'vdn':
        return('kg/m**3')
    if key == 'vpr':
        return('unitless')
    else:
        return('"'+key+'" is not a constant or function in this module.')



  
