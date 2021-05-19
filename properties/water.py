# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, water.py, is a python library of the                          #
# thermophysical properties of liquid water.  The properties, both         #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# water.py is distributed in the hope that it will be useful,              #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with thermoproperties.py.  If not, see                             #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this library, or other property data   #
# from the DIPPR(R) database, should include the citation below.           #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# ======================================================================== #
# water.py                                                                 #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - February 2018                                              #
# Version 1.1 - October 2019 Minor corrections to documentation of tsat.   #
# Version 1.2 - February 2020 Added docstring and unit function            #
# Version 1.3 - Changed name from waterproperties to water and added the   #
#               module to byutpl package. Changed pressure-dependent       #
#               functions to use Soave-Redlich-Kwong equation of state for #
#               the vapor density.                                         #
# ======================================================================== #
"""
This library contains functions for the properties of water.
The values come from the DIPPR(R) Sample database [1], 
and the DIPPR(R) abbreviations are used. The saturated vapor density
comes from solving the Soave-Redlich-Kwong equation of state.

This module is part of the byutpl package. Import the module using

  import byutpl.properties.water as water

When imported in this way, the properties can be called as   

  water.tc
  
for constant properties like critical temperature or
  
  water.vtc(t)

for temperature-dependent properties like vapor thermal conductivity
where `t` is temperature in units of K.
    
A complete list of properties, and the associated units, are found       
below.                                                                   

Function    Return Value                             Input Value         
---------   --------------------------------------   ----------------- 
tc          critical temperature in K                none              
pc          critical pressure in Pa                  none              
vc          critical volume in m**3/mol              none              
zc          critical compressibility factor          none              
mw          molecular weight in kg/mol               none              
acen        acentric factor                          none              
ldn(t)      liquid density in kg/m**3                temperature in K  
lcp(t)      liquid heat capacity in J/mol/K          temperature in K  
ltc(t)      liquid thermal conductivity in W/m/K     temperature in K  
vp(t)       liquid vapor pressure in Pa              temperature in K  
hvp(t)      heat of vaporization in J/mol            temperature in K  
pr(t)       Prandtl number                           temperature in K  
lvs(t)      liquid viscosity in Pa*s                 temperature in K  
nu(t)       liquid kinematic viscosity in m**2/s     temperature in K  
tsat(p)     temperature at saturation in K           pressure in Pa    
vvs(t)      vapor (steam) viscosity in Pa*s          temperature in K  
vtc(t)      vapor (steam) therm. conductiv. in W/m/K temperature in K  
vdnsat(t)   vapor (steam) sat. density in kg/m**3    temperature in K  

References
----------
.. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
   DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
   for Physical Properties, AIChE, New York, NY (2017).
"""

import numpy as np
from scipy.optimize import fsolve
from scipy import interpolate
import byutpl.eos.srk as srk
import byutpl.equations.dippreqns as dippr

# critical temperature
tc = 647.096 # units of K

# critical pressure
pc = 2.20640E7 # units of Pa

# critical volume
vc = 0.0000559472 # units of m**3/mol

# critical compressibility factor
zc = 0.229 # unitless

# acentric factor
acen = 0.344861 # unitless

# molecular weight
mw = 0.01801528 # units of kg/mol
  
def ldn(t):
    """Liquid density of water 
	
    Liquid density of water from the DIPPR(R) correlation.
    (Correlation C: DIPPR Equation 119; valid from 273.16 - 647.096 K;
    uncertainty: < 0.2%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density of water

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.7874E+01, 3.5618E+01, 1.9655E+01, -9.1306E+00, \
                  -3.1367E+01, -8.1356E+02, -1.7421E+07])
    tr = t/tc
    x = 1.0-tr
    y = dippr.eq119(x,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """Liquid heat capacity of water 
	
    Liquid heat capacity of water from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 100; valid from 273.16 - 533.15 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity of water

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([2.7637E+05, -2.0901E+03, 8.1250E+00, -1.4116E-02, 9.3701E-06])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """Liquid thermal conductivity of water 
	
    Liquid thermal conductivity of water from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 100; valid from 273.16 - 633.15 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal conductivity of water

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([-4.3200E-01, 5.7255E-03, -8.0780E-06, 1.8610E-09, 0])
    y = dippr.eq100(t,c)
    return(y)

def vp(t):
    """Liquid vapor pressure of water 
	
    Liquid vapor pressure of water from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 273.16 - 647.096 K;
    uncertainty: < 0.2%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of water

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([7.3649E+01, -7.2582E+03, -7.3037E+00, 4.1653E-06, 2.0])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t):
    """Heat of vaporization of water 
	
    Heat of vaporization of water from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 106; valid from 273.16 - 647.096 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat of vaporization of water

    Returns
    -------
    float
        The heat of vaporization (J/mol) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([5.6600E+07, 6.12041E-01, -6.25697E-01, 3.98804E-01, 0])
    tr = t/tc
    y = dippr.eq106(tr,c)
    y = y / 1000 # convert from J/kmol to J/mol
    return(y)
    
def lvs(t): # liquid viscosity
    r"""Liquid viscosity of water 
	
    Liquid viscosity of water from the DIPPR(R) correlation
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid viscosity of water

    Returns
    -------
    float
        The liquid viscosity (Pa*s) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = -5.2843E+01
    B = 3.7036E+03
    C = 5.8660E+00
    D = -5.8790E-29
    E = 10.0000E+00
    y = np.exp(A + B / t + C * np.log(t) + D * t**E)
    return y # units of Pa*s

def nu(t):
    r"""Liquid kinematic viscosity of water 
	
    Liquid kinematic viscosity of water calculated from the lvs and ldn functions
    in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid kinematic viscosity of water

    Returns
    -------
    float
        The liquid kinematic viscosity (m**2/s) of water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return lvs(t)/ldn(t) # m**2/s

def pr(t):
    r"""Prandtl number of liquid water 
	
    Prandtl number of liquid water calculated from the lcp, lvs, and ltc functions
    in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of liquid water

    Returns
    -------
    float
        The Prandtl number (dimensionless) of liquid water at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return lcp(t)*lvs(t)/ltc(t)/mw

def ftsat(t,p):
    r"""Function supplied to fsolve in tsat function 
	
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
    return vp(t) - p

def tsat(p):
    r"""Saturated temperature for water
	
    Saturation temperature of water for a given pressure 'p'.  It is
    the temperature for which the following equation is true:
    vp(t)= `p`
    where vp is the function in this module and t is the value
    this function (tsat) returns.
	
    Parameters
    ----------
    p : float
        The pressure (Pa) at which to find the saturated temperature

    Returns
    -------
    float
        The temperature (K) of water at saturation at pressure `p`.

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
    r"""Viscosity of vaporized water (steam) 
	
    Vapor viscosity of water (the viscosity of steam) at temperature `t`
    from the DIPPR(R) correlation.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor viscosity of water
    	(the viscosity of steam)

    Returns
    -------
    float
        The vapor viscosity of water (Pa*s) (the viscosity of steam) at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = 1.7096E-08
    B = 1.1146
    return A*t**B

def vtc(t):
    r"""Thermal conductivity of vaporized water (steam) 
	
    The vapor thermal conductivity of water (the thermal conductivity of steam)
    at temperature `t` from the DIPPR(R) correlation.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor thermal conductivity of water
    	(the thermal conductivity of steam)

    Returns
    -------
    float
        The vapor thermal conductivity of water (W m**-1 K**-1) 
    	(the thermal conductivity of steam) at `t`

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = 6.2041E-06
    B = 1.1146
    return A*t**B

def vdnsat(t):
    r"""Vapor density of saturated water (saturated steam) 
	
    The saturated vapor density of water (density of saturated steam)
    at temperature `t`.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the saturated vapor density of water
    	(the density of saturated steam)

    Returns
    -------
    float
        The saturated vapor density of water (kg/m**3) (the density of steam) at `t`

    References
    ----------
    .. T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""   
    tdata = [273.15,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,373.15,375,380,385,390,400,410,420,430]
    ddata = [206.3,181.7,130.4,99.4,69.7,51.94,39.13,29.74,22.93,17.82,13.98,11.06,8.82,7.09,5.74,4.683,3.846,3.18,2.645,2.212,1.861,1.679,1.574,1.337,1.142,0.98,0.731,0.553,0.425,0.331]
    tck = interpolate.splrep(tdata,ddata)
    y=interpolate.splev(t,tck)
    return 1.0/y

def unit(key):
    r"""Returns the units of `key` 
	
    Returns the units of the constant or function in this module identified
    by 'key'
	
    Parameters
    ----------
    key : string
        The name of the constant or function in this module for which the units are needed.

    Returns
    -------
    string
        The units for the constant or function identified by `key`

    References
    ----------
    .. T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
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
    if key == 'ltc':
        return('W m**-1 K**-1')
    if key == 'vp':
        return('Pa')
    if key == 'hvp':
        return('J/mol')
    if key == 'pr':
        return('unitless')
    if key == 'lvs':
        return('Pa*s')
    if key == 'nu':
        return('m**2/s')
    if key == 'tsat':
        return('K')
    if key == 'vvs':
        return('Pa*s')
    if key == 'vtc':
        return('W m**-1 K**-1')
    if key == 'vdnsat':
        return('kg/m**3')
    else:
        return('"'+key+'" is not a constant or function in this module.')
  
