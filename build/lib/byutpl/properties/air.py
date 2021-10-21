# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, air.py, is a python module of the                             #
# thermophysical properties of air.  The properties are taken from tables  #
# found in the two textbooks listed in the References section below.       #
# The data in these tables are cited from older references, and            #
# more accurate properties values are available for other sources such as  #
# DIPPR or NIST REFPROP. This module is intended for pedagogical purposes  #
# only.                                                                    #
#                                                                          #
# air.py is distributed in the hope that it will be useful,                #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     #
#                                                                          #
# All published work which utilizes this module should cite the            #
# works given in the References section below.                             #
#                                                                          #
# ======================================================================== #
# air.py                                                                   #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - February 2018                                              #
# Version 1.1 - February 2020 Added docstring and unit function            #
# Version 2.0 - May 2021 Changed name from airproperties to air and        #
#               and added the module to the byutpl package. Changed        #
#               pressure-dependent functions to use the Soave-Redlich-     #
#               Kwong equation of state. Changed the source of the data    #
#               from DIPPR to textbooks as air is not a compound in the    #
#               Sample (freely available) version of the DIPPR database.   #
# ======================================================================== #

"""
This library contains functions for the properties of air.
Pressure-dependent properties use the Soave-Redlich-Kwong equation of 
state when needed.

Import the module using

  import byutpl.properties.air as air

When imported in this way, the constant properties can be called as   

  air.mw
  
which returns the molecular weight. Temperature dependent properties
can be called as
  
  air.vtc(t)

which returns the vapor thermal conductivity at `t` where `t` is 
temperature in units of K. Temperature and pressure dependent properties
can be called as
  
  air.vcp(t,p)
    
which returns the vapor heat capacity at `t` and `p` where 
`t` is temperature in units of K and `p` is pressure in units of Pa.

(The "vapor" designation is meaningless as air is defined as a vapor, 
but other compounds in the byutpl package have "vapor" and "liquid"
properties. The "vapor" nomenclature is kept for consistency 
with these other compounds.)

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
icp(t)      ideal gas heat capacity in J/(mol*K)     temperature in K    
vtc(t)      vapor thermal conductivity in W/(m*K)    temperature in K    
vvs(t)      vapor viscosity in Pa*s                  temperature in K    
vdn(t,p)    vapor (steam) density in kg/m**3         temperature in K
                                                     pressure in Pa
vcp(t,p)    vapor isobaric heat capacity in          temperature in K
            J/(mol*K)                                pressure in Pa
vnu(t,p)    vapor kinematic viscosity in m**2/s      temperature in K
            m**2/s                                   pressure in Pa                                                     
vpr(t,p)    vapor Prandtl number (unitless)          temperature in K
                                                     pressure in Pa
valpha(t,p) vapor thermal diffusivity in m**2/s      temperature in K
                                                     pressure in Pa                                                    

Values for tc, pc, vc, zc, mw, acen, icp, and vcp come from [1]. icp 
is from the reported correlation in the reference. vcp is obtained from
icp and the residual heat capacity calculated from the Soave-Redlich-
Kwong equation of state. Values for vtc and vvs are calculated from
cubic splines of the tabular data found in [2]. vdn is from the 
Soave-Redlich-Kwong equation of state. vnu, valpha, and vpr are
obtained from the other properties.

The data in the tables in the textbooks are from older references, and
more accurate properties values are available for other sources such as
DIPPR or NIST REFPROP. This module is intended for pedagogical purposes
only.

References
----------
.. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
   Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
   New York, NY (2005).

.. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
   Fundamental of Heat and Mass Transfer 7th edition,
   John Wiley & Sons Inc., Hoboken, NJ (2011).
"""

import numpy as np
import byutpl.eos.srk as srk
from scipy import interpolate

# critical temperature
tc = 132.2 # units of K

# critical pressure
pc = 37.45e5 # units of Pa

# critical volume
vc = 0.0000848 # units of m**3/mol

# critical compressibility factor
zc = 0.289 # unitless

# acentric factor
acen = 0.035 # unitless

# molecular weight
mw = 0.028851 # units of kg/mol
  
def icp(t):
    """ideal gas heat capacity of air

    Ideal gas heat capacity of air from Smith, Van Ness, and Abbott [1].

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat capacity
        of air.

    Returns
    -------
    float
        The ideal gas heat capacity (J mol**-1 K**-1) of air at `t`.

    References
    ----------
    .. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
       Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
       New York, NY (2005).
	"""
    A = 3.355
    B = 0.575E-03
    C = 0.0
    D = -0.016E5
    y = A + B*t + C*t**2 + D*t**-2
    y = y * srk.rg
    return y

def vtc(t):
    """vapor thermal conductivity of air

    Thermal conductivity of air from a cubic spline of the tabular 
    data in Bergman, Lavine, Incropera, and Dewitt [1].

    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
    
    (valid from 100 - 3000 K; predicted uncertainty: >5%)

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the thermal conductivity 
        of air.

    Returns
    -------
    float
        The thermal conductivity (W m**-1 K**-1) of air at `t`.

    References
    ----------
    .. [1] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    DT=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750, \
                 800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700, \
                 1800,1900,2000,2100,2200,2300,2400,2500,3000])
    DP=np.array([9.34,13.8,18.1,22.3,26.3,30.0,33.8,37.3,40.7,43.9,46.9, \
                 49.7,52.4,54.9,57.3,59.6,62.0,64.3,66.7,71.5,76.3,82,91, \
                 100,106,113,120,128,137,147,160,175,196,222,486])
    tck = interpolate.splrep(DT,DP)
    y=interpolate.splev(t,tck)*1E-3 # table reports vtc*1000; removed here
    return y

def vvs(t):
    """vapor viscosity of air

    Vapor viscosity of air from a cubic spline of the tabular data in
    Bergman, Lavine, Incropera, and Dewitt [1].

    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
    
    (valid from 100 - 3000 K; predicted uncertainty: >10%)

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the viscosity of air. 

    Returns
    -------
    float
        The viscosity (Pa*s) of air at `t`.

    References
    ----------
    .. [1] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    DT=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750, \
                 800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700, \
                 1800,1900,2000,2100,2200,2300,2400,2500,3000])
    DP=np.array([71.1,103.4,132.5,159.6,184.6,208.2,230.1,250.7,270.1, \
                 288.4,305.8,322.5,338.8,354.6,369.8,384.3,398.1,411.3, \
                 424.4,449,473,496,530,557,584,611,637,663,689,715,740, \
                 766,792,818,955])
    tck = interpolate.splrep(DT,DP)
    y=interpolate.splev(t,tck)*1E-7 # table reports vtc*1E7; removed here
    return y

def vdn(t,p):
    """vapor density of air
	
    The vapor density of air at temperature `t` and pressure `p` from 
    the Soave-Redlich-Kwong equation of state.
    
    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
    
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor density of air.

    p : float
        The pressure (Pa) at which to evaluate the vapor density of air.

    Returns
    -------
    float
        The vapor density of air (kg/m**3) at `t` and `p`.
    """   
    v = srk.vv(t,p,tc,pc,acen)
    v = v / mw # convert from m**3/mol to m**3/kg
    return(1/v)

def vnu(t,p):
    """vapor kinematic viscosity of air
	
    Vapor kinematic viscosity of air calculated from the vvs and vdn 
    functions in this module.
    
    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic 
        viscosity of air.

    p : float
        The pressure (Pa) at which to evaluate the kinematic viscosity
        of air.

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of air at `t` and `p`.
       
    References
    ----------
    .. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
       Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
       New York, NY (2005).
       
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return(vvs(t)/vdn(t,p))

def valpha(t, p):
    """vapor thermal diffusivity of air

    Vapor thermal diffusivity of air calculated from the vtc, vdn, and icp 
    functions in this module. 
    
    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the thermal diffusivity 
        of air
   
    Returns
    -------
    float
        The thermal diffusivity (m**2/s) of air at `t` and `p`.

    References
    ----------
    .. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
       Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
       New York, NY (2005).
       
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return vtc(t)/vdn(t,p)*mw/icp(t)

def vpr(t,p):
    """vapor Prandtl number of air

    Vapor Prandtl number of air calculated from the vnu and valpha
    functions in this module.

    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number
        of air.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of air at `t` and `p`.

    References
    ----------
    .. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
       Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
       New York, NY (2005).
       
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return vnu(t,p)/valpha(t,p)

def vcp(t,p):
    """vapor heat capacity air
	
    Vapor capacity of air calculated from icp and the residual property
    from the Soave-Redlich-Kwong equation of state.

    (The "vapor" designation is meaningless as air is defined as a vapor, 
    but other compounds in the byutpl package have "vapor" and "liquid"
    properties. The "vapor" nomenclature is kept for consistency 
    with these other compounds.) 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat capacity 
        of air.

    p : float
        The pressure (Pa) at which to evaluate the heat capacity
        of air.

    Returns
    -------
    float
        The heat capacity (J/(mol*K)) of air at `t` and `p`.

    References
    ----------
    .. [1] J. M. Smith, H. C. Van Ness, M. M Abbott, Introduction to 
       Chemical Engineering Thermodynamics 7th edition, McGraw-Hill, 
       New York, NY (2005).
	"""
    x = icp(t) + srk.cprv(t,p,tc,pc,acen)
    return(x)    

def unit(key): # returns the units of key
    """returns the units of `key` 
	
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
    if key == 'icp':
        return('J mol**-1 K**-1')
    if key == 'vcp':
        return('J mol**-1 K**-1')
    if key == 'vtc':
        return('W m**-1 K**-1')
    if key == 'vvs':
        return('Pa*s')
    if key == 'vdn':
        return('kg/m**3')
    if key == 'vnu':
        return('m**2/s')
    if key == 'valpha':
        return('m**2/s')
    if key == 'vpr':
        return('unitless')
    else:
        return('"'+key+'" is not a constant or function in this module.')

  

    

  
