# Copyright (C) 2024 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, r134a.py, is a python module of the                           #
# thermophysical properties of R-134a.  The properties, both               #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# r134a.py is distributed in the hope that it will be useful,              #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with r134a.py.  If not, see                                        #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this module, or other property data    #
# from the DIPPR(R) database, should include the citation below.           #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# ======================================================================== #
# r134a.py                                                                 #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - April 2024                                                 #
# ======================================================================== #
"""
This library contains functions for the properties of 
1,1,1,2-tetrafluoroethane, commonly class R-134a.
The values come from the DIPPR(R) Public database [1], 
and the DIPPR(R) abbreviations are used. Vapor properties that are 
dependent on pressure are obtained using the Soave-Redlich-Kwong equation
of state.

Import the module using

  import byutpl.properties.r134a as r134a

When imported in this way, constant properties can be called as   

  r134a.acen
  
which returns the acentric factor. Temperature dependent properties
can be called as
  
  r134a.vtc(t)

which returns the vapor thermal conductivity at `t` where `t` is 
temperature in units of K. Temperature and pressure dependent properties
can be called as
  
  r134a.vcp(t,p)

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
tc = 374.18 # units of K

# critical pressure
pc = 4056000 # units of Pa

# critical volume
vc = 0.1988 # units of m**3/mol

# critical compressibility factor
zc = 0.259 # unitless

# acentric factor
acen = 0.326878 # unitless

# molecular weight
mw = 0.10203089 # units of kg/mol
  
def ldn(t):
    """liquid density of R-134a 
	
    Liquid density of R-134a from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 105; valid from 169.85 - 374.18 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density
        of r134a.

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.31, 0.26206, 374.18, 0.27779])
    y = dippr.eq105(t,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """liquid heat capacity of R-134a 
	
    Liquid heat capacity of R-134a from the DIPPR(R) correlation
    (Correlation B: DIPPR Equation 100; valid from 169.85 - 353.15 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity
        of R-134a.

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of R-134a at
        `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([651080, -9505.7, 62.835, -0.18264, 0.00020031])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """liquid thermal conductivity of R-134a 
	
    Liquid thermal conductivity of R-134a from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 100; valid from 169.85 - 360 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal
        conductivity of R-134a.

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([0.21973, -0.00046625, 1.5809E-08, 0, 0])
    y = dippr.eq100(t,c)
    return(y)

def vp(t):
    """liquid vapor pressure of R-134a 
	
    Liquid vapor pressure of R-134a from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 169.85 — 374.18 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of
        R-134a.

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([81.808, -4676.6, -9.4881, 1.5122E-05, 2])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t):
    """heat of vaporization of R-134a 
	
    Heat of vaporization of R-134a from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 106; valid from 169.85 — 374.18 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat of vaporization of
        R-134a.

    Returns
    -------
    float
        The heat of vaporization (J/mol) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([40274000,1.1232,-1.2239,0.5411,0.0])
    tr = t/tc
    y = dippr.eq106(tr,c)
    y = y / 1000 # convert from J/kmol to J/mol
    return(y)
    
def lvs(t):
    """liquid viscosity of R-134a 
	
    Liquid viscosity of R-134a from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 169.85 - 343.15 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid viscosity of
        R-134a.

    Returns
    -------
    float
        The liquid viscosity (Pa*s) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([26.746,-649.07,-5.8056,0,0])
    y = dippr.eq101(t,c)
    return(y)

def lnu(t):
    """liquid kinematic viscosity of R-134a 
	
    Liquid kinematic viscosity of R-134a calculated from the lvs and ldn
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid kinematic 
        viscosity of R-134a.

    Returns
    -------
    float
        The liquid kinematic viscosity (m**2/s) of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(lvs(t)/ldn(t))

def lpr(t):
    """Prandtl number of liquid R-134a 
	
    Prandtl number of liquid R-134a calculated from the lcp, lvs, and ltc
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of
        liquid R-134a.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of liquid R-134a at `t`.

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
    """saturated temperature for R-134a
	
    Saturation temperature of R-134a for a given pressure 'p'.  It is
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
        The temperature (K) of R-134a at saturation at pressure `p`.

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
    """vapor viscosity of R-134a
	
    Vapor viscosity of R-134a at temperature `t` from the DIPPR(R)
    correlation.
    (Correlation A: DIPPR Equation 102; valid from 169.85 — 1000 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor viscosity of
        R-134a.

    Returns
    -------
    float
        The vapor viscosity of R-134a at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.341E-06,0.50123,287.59,0])
    y = dippr.eq102(t,c)
    return(y)

def vtc(t):
    """vapor thermal conductivity of R-134a
	
    The vapor thermal conductivity of R-134a at temperature `t`
    from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 102; valid from 169.85 — 1000 K;
    uncertainty: < 10%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor thermal
        conductivity of R-134a.

    Returns
    -------
    float
        The vapor thermal conductivity of R-134a (W m**-1 K**-1) 
        at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([7.5788E-03,0.50807,2178.4,1.85490E+05])
    y = dippr.eq102(t,c)
    return(y)

def vdn(t,p):
    """vapor density of R-134a
	
    The vapor density of R-134a at temperature `t` and
    pressure `p` from the Soave-Redlich-Kwong equation of state. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor density of
        R-134a.

    p : float
        The pressure (Pa) at which to evaluate the vapor density of
        R-134a.

    Returns
    -------
    float
        The vapor density of R-134a (kg/m**3) at `t` and `p`.
	"""   
    v = srk.vv(t,p,tc,pc,acen)
    v = v / mw # convert from m**3/mol to m**3/kg
    return(1/v)
    
def icp(t): 
    """ideal gas heat capacity of R-134a
    
    The ideal gas heat capacity of R-134a at temperature `t` from the 
    DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 127; valid from 20 — 1500 K;
    uncertainty: < 1%)
    
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat
        capacity of R-134a.

    Returns
    -------
    float
        The ideal gas heat capacity of R-134a (J/(mol*K)) at `t`.
    """     
    c=np.array([33257.8886,42635.713824,413.065128,85068.10172,1459.742655,16896.778664,3859.353771])    
    y=dippr.eq127(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def vcp(t,p):
    """vapor heat capacity R-134a
	
    Heat capacity of vapor R-134a calculated from the DIPPR(R) 
    correlation for ideal gas heat capacity and the residual property
    from the Soave-Redlich-Kwong equation of state.
    (ICP Correlation A: DIPPR Equation 127; valid from 20 — 1500 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat 
        capacity of vapor R-134a.

    p : float
        The pressure (Pa) at which to evaluate the heat capacity
        of vapor R-134a.

    Returns
    -------
    float
        The heat capacity (J/(mol*K)) of vapor R-134a
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
    """vapor kinematic viscosity of R-134a
	
    Kinematic viscosity of vapor R-134a calculated from the vvs and
    vdn functions in this module. The calculation uses the Soave-Redlich-
    Kwong equation of state for the vapor density.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic 
        viscosity of vapor R-134a.

    p : float
        The pressure (Pa) at which to evaluate the kinematic viscosity
        of vapor R-134a.

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of vapor R-134a
        at `t` and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vvs(t)/vdn(t,p))

def vpr(t, p):
    """Prandtl number of vapor R-134a
	
    Prandtl number of vapor R-134a calculated from the vcp, vvs, 
    and vtc functions in this module. The calculation uses the Soave-
    Redlich-Kwong equation of state to correct the ideal gas heat capacity
    to the real gas at `t` and `p`. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of vapor
        R-134a.

    p : float
        The pressure (Pa) at which to evaluate the Prandtl number of vapor
        R-134a.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of vapor R-134a at `t`
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



  
