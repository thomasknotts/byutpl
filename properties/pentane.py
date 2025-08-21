# Copyright (C) 2025 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, pentane.py, is a python module of the                         #
# thermophysical properties of pentane.  The properties, both              #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# pentane.py is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with pentane.py.  If not, see                                      #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this module, or other property data    #
# from the DIPPR(R) database, should include the citation below.           #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# ======================================================================== #
# pentane.py                                                               #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - August 2025                                                #
# ======================================================================== #
"""
This library contains functions for the properties of pentane.
The values come from the DIPPR(R) Sample database [1], 
and the DIPPR(R) abbreviations are used. Vapor properties that are 
dependent on pressure are obtained using the Soave-Redlich-Kwong equation
of state.

This module is part of the byutpl package. Import the module using

  import byutpl.properties.pentane as pen

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
.. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
   DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
   for Physical Properties, AIChE, New York, NY (2017).
"""

import numpy as np
from scipy.optimize import fsolve
import byutpl.eos.srk as srk
import byutpl.equations.dippreqns as dippr

# critical temperature
tc = 469.7 # units of K

# critical pressure
pc = 3.37e6 # units of Pa

# critical volume
vc = 0.000313 # units of m**3/mol

# critical compressibility factor
zc = 0.27 # unitless

# acentric factor
acen = 0.251506 # unitless

# molecular weight
mw = 72.14878e-3 # units of kg/mol
  
def ldn(t):
    """liquid density of pentane 
	
    Liquid density of pentane from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 105; valid from 143.42 - 469.7 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density
        of pentane.

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([0.84947,0.26726,469.7, 0.27789])
    y = dippr.eq105(t,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """liquid heat capacity of pentane 
	
    Liquid heat capacity of pentane from the DIPPR(R) correlation
    (Correlation B: DIPPR Equation 100; valid from 143.42 - 390.00 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity
        of pentane.

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of pentane at
        `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([159080,-270.5,0.99537,0,0])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """liquid thermal conductivity of pentane 
	
    Liquid thermal conductivity of pentane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 123; valid from 143.42 - 445 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal
        conductivity of pentane.

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([0.0775839,-0.2486,-1.77869,4.18476])
    tau=1-t/tc
    y = dippr.eq123(tau,c)
    return(y)

def vp(t):
    """liquid vapor pressure of pentane 
	
    Liquid vapor pressure of pentane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 143.42 — 469.7 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of
        pentane.

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([78.741,-5420.3,-8.8253,9.6171E-06,2])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t):
    """heat of vaporization of pentane 
	
    Heat of vaporization of pentane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 106; valid from 143.42 — 469.7 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat of vaporization of
        pentane.

    Returns
    -------
    float
        The heat of vaporization (J/mol) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([45087000,0.95886,-0.92384,0.39393,0])
    tr = t/tc
    y = dippr.eq106(tr,c)
    y = y / 1000 # convert from J/kmol to J/mol
    return(y)
    
def lvs(t):
    """liquid viscosity of pentane 
	
    Liquid viscosity of pentane from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 143.42 - 465.15 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid viscosity of
        pentane.

    Returns
    -------
    float
        The liquid viscosity (Pa*s) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([-53.509,1836.6,7.1409,-1.9627E-05,2])
    y = dippr.eq101(t,c)
    return(y)

def lnu(t):
    """liquid kinematic viscosity of pentane 
	
    Liquid kinematic viscosity of pentane calculated from the lvs and ldn
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid kinematic 
        viscosity of pentane.

    Returns
    -------
    float
        The liquid kinematic viscosity (m**2/s) of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(lvs(t)/ldn(t))

def lpr(t):
    """Prandtl number of liquid pentane 
	
    Prandtl number of liquid pentane calculated from the lcp, lvs, and ltc
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of
        liquid pentane.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of liquid pentane at `t`.

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

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vp(t) - p)

def tsat(p):
    """saturated temperature for pentane
	
    Saturation temperature of pentane for a given pressure 'p'.  It is
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
        The temperature (K) of pentane at saturation at pressure `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    x = 700 # guess in K
    y = fsolve(ftsat,x,p)
    return(y[0])
    
def vvs(t):
    """vapor viscosity of pentane
	
    Vapor viscosity of pentane at temperature `t` from the DIPPR(R)
    correlation.
    (Correlation A: DIPPR Equation 102; valid from 143.42 — 1000 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor viscosity of
        pentane.

    Returns
    -------
    float
        The vapor viscosity of pentane at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([6.3412E-08,0.84758,41.718,0.0])
    y = dippr.eq102(t,c)
    return(y)

def vtc(t):
    """vapor thermal conductivity of pentane
	
    The vapor thermal conductivity of pentane at temperature `t`
    from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 102; valid from 273.15 — 1000 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor thermal
        conductivity of pentane.

    Returns
    -------
    float
        The vapor thermal conductivity of pentane (W m**-1 K**-1) 
        at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([-684.4,0.764,-1055000000,0])
    y = dippr.eq102(t,c)
    return(y)

def vdn(t,p):
    """vapor density of pentane
	
    The vapor density of pentane at temperature `t` and
    pressure `p` from the Soave-Redlich-Kwong equation of state. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor density of
        pentane.

    p : float
        The pressure (Pa) at which to evaluate the vapor density of
        pentane.

    Returns
    -------
    float
        The vapor density of pentane (kg/m**3) at `t` and `p`.
	"""   
    v = srk.vv(t,p,tc,pc,acen)
    v = v / mw # convert from m**3/mol to m**3/kg
    return(1/v)
    
def icp(t): 
    """ideal gas heat capacity of pentane
    
    The ideal gas heat capacity of pentane at temperature `t` from the 
    DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 107; valid from 200 — 1500 K;
    uncertainty: < 1%)
    
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat
        capacity of pentane.

    Returns
    -------
    float
        The ideal gas heat capacity of pentane (J/(mol*K)) at `t`.
    """     
    c=np.array([88050,301100,1650.2,189200,747.6])
    y=dippr.eq107(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def vcp(t,p):
    """vapor heat capacity pentane
	
    Heat capacity of vapor pentane calculated from the DIPPR(R) 
    correlation for ideal gas heat capacity and the residual property
    from the Soave-Redlich-Kwong equation of state.
    (ICP Correlation A: DIPPR Equation 127; valid from 20 — 1500 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat 
        capacity of vapor pentane.

    p : float
        The pressure (Pa) at which to evaluate the heat capacity
        of vapor pentane.

    Returns
    -------
    float
        The heat capacity (J/(mol*K)) of vapor pentane
        at `t` and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    x = icp(t) + srk.cprv(t,p,tc,pc,acen)
    return(x)

def vnu(t,p):
    """vapor kinematic viscosity of pentane
	
    Kinematic viscosity of vapor pentane calculated from the vvs and
    vdn functions in this module. The calculation uses the Soave-Redlich-
    Kwong equation of state for the vapor density.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic 
        viscosity of vapor pentane.

    p : float
        The pressure (Pa) at which to evaluate the kinematic viscosity
        of vapor pentane.

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of vapor pentane
        at `t` and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vvs(t)/vdn(t,p))

def vpr(t, p):
    """Prandtl number of vapor pentane
	
    Prandtl number of vapor pentane calculated from the vcp, vvs, 
    and vtc functions in this module. The calculation uses the Soave-
    Redlich-Kwong equation of state to correct the ideal gas heat capacity
    to the real gas at `t` and `p`. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of vapor
        pentane.

    p : float
        The pressure (Pa) at which to evaluate the Prandtl number of vapor
        pentane.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of vapor pentane at `t`
        and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
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

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
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
    if key == 'lcp':
        return('J mol**-1 K**-1')
    if key == 'icp':
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



  
