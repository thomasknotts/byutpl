# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, air.py, is a python library of the                            #
# thermophysical properties of air.  The properties that are pressure-     #
# independent, both constant and temperature-dependent, are taken from the #
# DIPPR(R) Public database.  A such, it can only be used while a student   #
# at BYU.                                                                  #
#                                                                          #
# air.py is distributed in the hope that it will be useful,                #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     #
#                                                                          #
# All published work which utilizes this library, or other property data   #
# from the DIPPR(R) database should have a Public or Sponsor license for   #
# the DIPPR(R) database and include the citation below.                    #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
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
#               pressure-dependent functions to use Soave-Redlich-Kwong    #
#               equation of state for real gas vapor heat capacity.        #
# ======================================================================== #

"""
This library contains functions for the properties of air.
Values for tc, pc, vc, zc, mw, acen, and icp(t) come from [1]. 
Values for  vtc, vvs, rho, nu, alpha, alpha, and pr crom from [2].

*****
This library should only be used while a student at BYU as it is from a 
version of the DIPPR database that requires a subscription.
*****    

Import the module using

  import airproperties as air

When imported in this way, the properties can be called as   

  air.tc
  
for constant properties like critical temperature or
  
  air.vtc(t)

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
icp(t)      ideal gass heat capacity in J/mol/K      temperature in K    
vtc(t)      vapor thermal conductivity in W/m/K      temperature in K    
vvs(t)      vapor viscosity in Pa*s                  temperature in K    
rho1atm(t)  density at 1 atm in kg/m**3              temperature in K    
nu1atm(t)   kinematic viscosity at 1 atm in m**2/s   temperature in K    
alpha1atm(t)thermal diffusivity at 1atm in m**2/s    temperature in K    
pr1atm(t)   Prandtl number at 1 atm                  temperature in K  

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
from scipy   import interpolate

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
  
def icp(t): # ideal gas heat capacity
    r"""Ideal gas heat capacity of air

    Ideal gas heat capacity of air from the DIPPR(R) correlation.

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat capacity of air

    Returns
    -------
    float
        The ideal gas heat capacity (J mol**-1 K**-1) of air at `t`

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = 2.8958E+04
    B = 9.3900E+03
    C = 3.0120E+03
    D = 7.5800E+03
    E = 1.4840E+03
    y = A+B*((C/t)/np.sinh(C/t))**2+D*((E/t)/np.cosh(E/t))**2
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return y # units of J/mol/K

def vtc(t): # thermal conductivity
    r"""Vapor thermal conductivity of air

    Thermal conductivity of air from the DIPPR(R) correlation.
    (The "vapor" part is meaningless for air since it only 
    exists as a vapor, but other compounds have "vapor" and 
    "liquid" thermal conductivities.  The "vapor" nomenclature remains 
    for consistency with libraries for other compounds.

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the thermal conductivity of air

    Returns
    -------
    float
        The thermal conductivity (W m**-1 K**-1) of air at `t`

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = 3.1417E-04
    B = 7.7860E-01
    C = -7.1160E-01
    D = 2.1217E+03
    y = A*t**B/(1+C/t+D/t**2)
    return y # units of W/m/K

def vvs(t): # vapor viscosity of air
    r"""Vapor viscosity of air

    Viscosity of air from the DIPPR(R) correlation.
    (The "vapor" part is meaningless for air since it only 
    exists as a vapor, but other compounds have "vapor" and 
    "liquid" thermal conductivities.  The "vapor" nomenclature remains 
    for consistency with libraries for other compounds.

    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the viscosity of air

    Returns
    -------
    float
        The viscosity (Pa*s) of air at `t`

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    A = 1.4250E-06
    B = 5.0390E-01
    C = 1.0830E+02
    D = 0
    y = A*t**B/(1+C/t+D/t**2)
    return y # Pa*s

def rho1atm(t): # density at 1 atm
    r"""Density of air at a pressure of 1 atm

    Density of air at a pressure of 1 atm and temperature `t`.
    The same information could be obtained from an appropriate
    equation of state.  The value returned is from a cubic spline of 
    the data from Table A.4 of Bergman et al.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the density of air for a pressure of 1 atm

    Returns
    -------
    float
        The density (kg/m**3) of air at 1 atm and `t`.

    References
    ----------
    .. T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    DT=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,3000])
    DR=np.array([3.5562,2.3364,1.7458,1.3947,1.1614,0.9950,0.8711,0.7740,0.6964,0.6329,0.5804,0.5356,0.4975,0.4643,0.4354,0.4097,0.3868,0.3666,0.3482,0.3166,0.2902,0.2679,0.2488,0.2322,0.2177,0.2049,0.1935,0.1833,0.1741,0.1658,0.1582,0.1513,0.1488,0.1389,0.1135])
    tck = interpolate.splrep(DT,DR)
    y=interpolate.splev(t,tck)
    return y # kg/m**3

def nu1atm(t): # kinetmatic viscosity at 1 atm
    r"""Kinematic viscosity of air at a pressure of 1 atm

    Kinematic viscosity of air at a pressure of 1 atm and temperature `t`.
    It is obtained using the vvs and rho1atm function in this module. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic viscosity of air
    	for a pressure of 1 atm

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of air at 1 atm and `t`.
		
    References
    ----------
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	   
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return vvs(t)/rho1atm(t) # m**2/s

def alpha1atm(t): # thermal diffusivity
    r"""Thermal diffusivity of air at a pressure of 1 atm

    Thermal diffusivity of air at a pressure of 1 atm and temperature `t`.
    It is obtained using the vtc, rho1atm, and icp functions in this module. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the thermal diffusivity of air
    	for a pressure of 1 atm

    Returns
    -------
    float
        The thermal diffusivity (m**2/s) of air at 1 atm and `t`.

    References
    ----------
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	   
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return vtc(t)/rho1atm(t)*mw/icp(t) # m**2/s

def pr1atm(t): # Prandtl Number
    r"""Prandtl number of air at a pressure of 1 atm

    Prandtl number of air at a pressure of 1 atm and temperature `t`.
    It is obtained using the nu1atm and alpha1atm functions in this module. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of air
    	for a pressure of 1 atm

    Returns
    -------
    float
        The Prandtl number (dimensionless) of air at 1 atm and `t`.

    References
    ----------
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	   
    .. [2] T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, 
       Fundamental of Heat and Mass Transfer 7th edition,
       John Wiley & Sons Inc., Hoboken, NJ (2011).
	"""
    return nu1atm(t)/alpha1atm(t) 

def unit(key): # returns the units of key
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
    if key == 'icp':
        return('J mol**-1 K**-1')
    if key == 'vtc':
        return('W m**-1 K**-1')
    if key == 'vvs':
        return('Pa*s')
    if key == 'rho1atm':
        return('kg/m**3')
    if key == 'nu1atm':
        return('m**2/s')
    if key == 'alpha1atm':
        return('m**2/s')
    if key == 'pr1atm':
        return('unitless')
    else:
        return('"'+key+'" is not a constant or function in this module.')

  

    

  
