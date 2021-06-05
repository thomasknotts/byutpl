# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, benzene.py, is a python module of the                         #
# thermophysical properties of benzene.  The properties, both              #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# benzene.py is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with benzene.py.  If not, see                                      #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this module, or other property data    #
# from the DIPPR(R) database, should include the citation below.           #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# ======================================================================== #
# benzene.py                                                               #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - February 2018                                              #
# Version 2.0 - May 2021 Changed name from benzeneproperties to benzene    #
#               and added the module to byutpl package. Changed            #
#               pressure-dependent functions to use Soave-Redlich-Kwong    #
#               equation of state for real gas vapor heat capacity. Added  #
#               vapor properties vdn, vcp, vnu, and vpr. Changed the name  #
#               of pr to lpr and nu to lnu. Removed vdn.                #
# ======================================================================== #
"""
This library contains functions for the properties of benzene.
The values come from the DIPPR(R) Sample database [1], 
and the DIPPR(R) abbreviations are used. Vapor properties that are 
dependent on pressure are obtained using the Soave-Redlich-Kwong equation
of state.

This module is part of the byutpl package. Import the module using

  import byutpl.properties.benzene as benzene

When imported in this way, constant properties can be called as   

  benzene.acen
  
which returns the acentric factor. Temperature dependent properties
can be called as
  
  benzene.vtc(t)

which returns the vapor thermal conductivity at `t` where `t` is 
temperature in units of K. Temperature and pressure dependent properties
can be called as
  
  benzene.vcp(t,p)

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
tc = 562.05 # units of K

# critical pressure
pc = 4.895e6 # units of Pa

# critical volume
vc = 0.000256 # units of m**3/mol

# critical compressibility factor
zc = 0.268 # unitless

# acentric factor
acen = 0.2103 # unitless

# molecular weight
mw = 0.07811184 # units of kg/mol
  
def ldn(t):
    """liquid density of benzene 
	
    Liquid density of benzene from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 105; valid from 278.68 - 562.05 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density
        of benzene.

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of benzene at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.0259, 0.26666, 562.05, 0.28394])
    y = dippr.eq105(t,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """liquid heat capacity of benzene 
	
    Liquid heat capacity of benzene from the DIPPR(R) correlation
    (Correlation B: DIPPR Equation 100; valid from 278.68 - 500.00 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity
        of benzene.

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of benzene at
        `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([162940, -344.94, 0.85562, 0, 0])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """liquid thermal conductivity of benzene 
	
    Liquid thermal conductivity of benzene from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 123; valid from 278.68 - 540.00 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal
        conductivity of benzene.

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of benzene at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([0.0542518, 2.74187, -7.22561, 8.22561])
    x = 1-t/tc
    y = dippr.eq123(x,c)
    return(y)

def vp(t):
    """liquid vapor pressure of benzene 
	
    Liquid vapor pressure of benzene from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 278.68 — 562.05 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of
        benzene.

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of benzene at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([83.107, -6486.2, -9.2194, 6.9844E-06, 2])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t): # heat of vaporization
    A = 50007000
    B = 0.65393
    C = -0.65393
    D = 0.029569
    tr = t/tc
    y = A * (1.0-tr)**(B + C * tr + D * tr**2)
    y = y / 1000 # convert from J/kmol to J/mol
    return y # J/mol
    
def lvs(t): # liquid viscosity
    A = 7.5117
    B = 294.68
    C = -2.794
    D = 0
    E = 0
    y = np.exp(A + B / t + C * np.log(t) + D * t**E)
    return y # units of Pa*s

def nu(t): # kinematic liquid viscosity
    return lvs(t)/ldn(t) # m**2/s

def lpr(t): # liquid Prandtl number
    return lcp(t)*lvs(t)/ltc(t)/mw # unitless

def ftsat(t,p): # function to calculate tsat with fsolve
    return vp(t) - p

def tsat(p): # saturation temperature (K) at pressure P (Pa)
    x = 700 # guess in K
    y = fsolve(ftsat,x,p)
    return(y[0]) # K
    
def vvs(t): # vapor viscosity
    A = 0.00000003134
    B = 0.9676
    C = 7.9
    return (A*t**B)/(1+C/t) # Pa*s

def vtc(t): # vapor thermal conductivity
    A = 0.00001652
    B = 1.3117
    C = 491
    return (A*t**B)/(1+C/t) # W/m/K   

def icp(t): # ideal gas heat capacity
    A = 33257.8886
    B = 51444.739266
    C = -761.088083
    D = 139737.490488
    E = 1616.907907
    F = 56829.10351
    G = 4111.398275
    y = A + B*(C/t)**2*np.exp(C/t)/(np.exp(C/t)-1)**2 + D*(E/t)**2*np.exp(E/t)/(np.exp(E/t)-1)**2 + F*(G/t)**2*np.exp(G/t)/(np.exp(G/t)-1)**2
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return y # units of J/mol/K

def vpr(t): # liquid Prandtl number
    return icp(t)*vvs(t)/vtc(t)/mw # unitless



  
