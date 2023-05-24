# Copyright (C) 2019 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, dippreqns.py, is a python module with functions that          #
# accept a temperature in K and an array of coefficients and return the    #
# value of the DIPPR equation.                                             #
#                                                                          #
# dippreqns.py is free software: you can redistribute it and/or            #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# dippreqns.py is distributed in the hope that it will be useful,          #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with dippreqns.py.  If not, see                                    #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# dippreqns.py                                                             #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - September 2019                                             #
# Version 2.0 - May 2021 Added docstring; Renamed from dippres.py to       #
#               dippreqns.py; Reorganized package structure                #
# Version 2.1 - October 2022 Changed functions to account for parameters   #
#               matrices that are not completely filled.                   #
# ======================================================================== #
"""
This module contains functions that accept a temperature or 
dimensionless temperature, and an array of coefficients,
and return the value of the DIPPR equation.      

The equations are numbered as outlined in the DIPPR Policies and 
Procedures Manual© [1] and are listed below.
  
The library can be loaded into python via the following command:         

  import dippreqns as dippr                                                

Loaded in this manner, the functions can be called with syntax like the 
following.

  dippr.eq100(t,c)

This would return the value of DIPPR Equation 100 at temperature `t` 
for the coefficients found in array `c`.

======================================================================   
INPUT PARAMETERS FOR FUNCTIONS                             
======================================================================   
Symbol                Property                           Units          
----------------------------------------------------------------------   
t                     system temperature                 K              
tr                    reduced system temperature (t/tc)  unitless       
tau                   1 - tr                             unitless       
c                     array of coefficients              DIPPR Default  
======================================================================   


======================================================================   
AVAILABLE FUNCTIONS                                             
======================================================================   
Function              Return Value                       Units          
----------------------------------------------------------------------   
eq100(t,c)            value of Eqn. 100 at t given c     DIPPR Default  
eq101(t,c)            value of Eqn. 101 at t given c     DIPPR Default  
eq102(t,c)            value of Eqn. 102 at t given c     DIPPR Default  
eq104(t,c)            value of Eqn. 104 at t given c     DIPPR Default  
eq105(t,c)            value of Eqn. 105 at t given c     DIPPR Default  
eq106(tr,c)           value of Eqn. 106 at tr given c    DIPPR Default  
eq107(t,c)            value of Eqn. 107 at t given c     DIPPR Default  
eq114(tau,c)          value of Eqn. 114 at tau given c   DIPPR Default  
eq115(t,c)            value of Eqn. 115 at t given c     DIPPR Default  
eq116(tau,c)          value of Eqn. 116 at tau given c   DIPPR Default   
eq119(tau,c)          value of Eqn. 119 at tau given c   DIPPR Default    
eq123(tau,c)          value of Eqn. 123 at tau given c   DIPPR Default   
eq124(tau,c)          value of Eqn. 124 at tau given c   DIPPR Default   
eq127(t,c)            value of Eqn. 127 at t govem c     DIPPR Default  
====================================================================== 

References
----------
.. [1] Policies and Procedures Manual for DIPPR Project 801, 
   American Institute of Chemical Engineerins, New York (1998).
"""
import numpy as np

# -------------------------------------------------------------------- #
# DIPPR Equations                                                      #
# -------------------------------------------------------------------- #
def fillcoeff(c):
    """Returns an Nx1 array that equals c and fills blanks.
       
    The DIPPR equations require a certain number of parameters.
    Some entries can be zero or blank and return the same value.
    This function returns an Nx1 array. If c contains less than 
    N entries, the returned array will have 0 in each of the 
    `N-len(c)` positions.
    
    Parameters
    ----------
    c : array of floats
        tempearture (K)
        
    N : length of Nx1 array needed.
    
    Returns
    -------
    7x1 array of floats
        The entries in `c` will be the first `len(c)` entries. 
        If `7-len(c) >0`, the remaining entries will be zero.
    """
    a=np.zeros(7)
    for i in range(len(c)): a[i]=c[i]
    return(a)

def eq100(t,c):
    """DIPPR Equation 100
       
    A fourth order polynomial
    Y = c[0] + c[1]*t + c[2}*t**2 + c[3]*t**3 + c[4}*t**4
    
    Properties: SDN (kmol/m**3), SCP (J/(kmol*K))
    Alternate eqn for: ICP (J/(kmol*K)), SVR (m**3/kmol), 
                       LDN (kmol/m**3), HVP (J/kmol), ST (N/m)
                       VTC (W/(m*K)), LTC (W/(m*K)), LCP (J/(kmol*K))
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 5x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : coefficient on t term
        c[2] : coefficient on t**2 term
        c[3] : coefficient on t**3 term
        c[4] : coefficient on t**4 term
    
    Returns
    -------
    float
        Value of DIPPR Equation 100 at `t` given `c`. 
        Units: default DIPPR units for property described by `c`.
    """
    a=fillcoeff(c)
    x = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4
    return(x) 
    
def eq101(t,c):
    """DIPPR Equation 101
    
    The Reidel equation
    Y = e**(c[0] + c[1]/t + c[2}*ln(t) + c[3}*t**c[4])
    
    Properties: VP (Pa), SVP (Pa), LVS (Pa*s)
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 5x1 array 
        the coefficients for the equation
        c[0] : constant in exponent
        c[1] : coefficient on 1/t term in exponent
        c[2] : coefficient on ln(t) term in exponent
        c[3] : coefficient on t**c[4] term in exponent
        c[4] : power on `t` for c[3] term in exponent
    
    Returns
    -------
    float
        Value of DIPPR Equation 101 at `t` given `c`. 
        Units: default DIPPR units for property described by `c`.
    """
    a=fillcoeff(c)
    x = np.exp(a[0] + a[1]/t + a[2]*np.log(t) + a[3]*t**a[4])
    return(x)

def eq101a(t,c):
    """DIPPR Equation 101a
    
    The temperature derivative of the natural log of the
    Reidel equation
    dlnY/dt = -c[1]/t**2 + c[2}/t + c[3}c[4]*t**(c[4]-1)
    It is usually used to obtain dVP/dt.
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 5x1 array 
        the coefficients for Equation 101; see documentation for `eq101`
    
    Returns
    -------
    float
        Value of temperature derivative of the natural logarithm
        of DIPPR Equation 101 at `t` given `c`. 
        Units: default DIPPR units for property described by `c`
        divided by K; for VP or SVP, Pa/K
    """
    a=fillcoeff(c)
    x = -1.0*a[1]/t**2 + a[2]/t + a[3]*a[4]*t**(a[4]-1.0)
    return(x)

def eq102(t,c):
    """DIPPR Equation 102
    
    Y = c[0]*t**c[1]/(1 + c[2]/t + c[3]/t**2)
    
    Properties: SCP (J/(kmol*K)), VVS (Pa*s), VTC (W/(m*K))
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 4x1 array 
        the coefficients for the equation
        c[0] : coefficient on t**c[1] term
        c[1] : power on `t` for c[0] term
        c[2] : coefficient on 1/t term
        c[3] : coefficient on 1/t**2 term
    
    Returns
    -------
    float
        Value of DIPPR Equation 102 at `t` given `c`. 
        Units: default DIPPR units for property described by `c`
    """
    a=fillcoeff(c)
    x = a[0]*t**a[1]/(1 + a[2]/t + a[3]/t**2)
    return(x)
  
def eq104(t,c):
    """DIPPR Equation 104
    
    Y = c[0] + c[1]/t + c[2]/t**3 + c[3]/t**8 + c[4]/t**9
    
    Property: SVR (m**3/kmol)
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 5x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : coefficient on 1/t term
        c[2] : coefficient on 1/t**3 term
        c[3] : coefficient on 1/t**8 term
        c[4] : coefficient on 1/t**9 term   
        
    Returns
    -------
    float
        Value of DIPPR Equation 104 at `t` given `c`. 
        Units: default DIPPR units for property described by `c` which
               should be m**3/mol
               
    """
    a=fillcoeff(c)
    x = a[0] + a[1]/t + a[2]/t**3 + a[3]/t**8 + a[4]/t**9
    return(x)
 
def eq105(t,c):
    """DIPPR Equation 105
    
    Rackett Equation
    Y = c[0]/c[1]**(1 + (1 - t/c[2])**c[3])
    
    Property: LDN (kmol/m**3)
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 4x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : base of power term
        c[2] : see equation above
        c[3] : see equation above
              
    Returns
    -------
    float
        Value of DIPPR Equation 105 at `t` given `c`. 
        Units: default DIPPR units for property described by `c`.
    """
    if np.count_nonzero(c) > 4:
        print("Error: DIPPR Equation 105 requires 4 or fewer nonzero parameters.")
        return(float("NaN"))
    x = c[0]/c[1]**(1 + (1 - t/c[2])**c[3])
    return(x)
  
def eq105a(t,c):
    """DIPPR Equation 105a
    
    The temperature derivative of Equation 105. This is not an official
    DIPPR equation. It is usually used to obtain dLDN/dt.
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 4x1 array 
        the coefficients for Equation 105, see documentation for `eq105`

    Returns
    -------
    float
        Value of temperature derivative of DIPPR Equation 105 at `t` 
        given `c`. 
        Units: default DIPPR units for property described by `c`
        divided by K. For LDN, kmol/(m**3*K)
    """
    if np.count_nonzero(c) != 4:
        print("Error: DIPPR Equation 105a requires exactly 4 nonzero parameters.")
        return(float("NaN"))
    x = eq105(t,c)*c[3]/c[2]*np.log(c[1])*(1 - t/c[2])**(c[3]-1)
    return(x)

def eq106(tr,c):
    """DIPPR Equation 106
    
    Y = c[0]*(1-tr)**(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)
    
    Property: HVP (J/kmol), ST (N/m)
    
    Parameters
    ----------
    tr : float
        reduced tempearture t/tc (unitless)
        
    c : 1x1 to 5x1 array 
        the coefficients for the equation
        c[0] : base of power term
        c[1] : constant in exponent
        c[2] : coefficient on tr term in exponent
        c[3] : coefficient on tr**2 term in exponent
        c[4] : coefficient on tr**3 term in exponent       
        
    Returns
    -------
    float
        Value of DIPPR Equation 106 at `tr` given `c`. 
        Units: default DIPPR units for property described by `c`.
    """
    a=fillcoeff(c)
    x = a[0]*(1-tr)**(a[1] + a[2]*tr + a[3]*tr**2 + a[4]*tr**3)
    return(x)

def eq106a(tr,tc,c):
    """DIPPR Equation 106a
    
    The temperature derivative of Equation 106. This is not an official
    DIPPR equation. It is usually used to obtain dHVP/dt.
    
    Parameters
    ----------
    tr : float
        reduced tempearture t/tc (unitless)
        
    tc : float
        critical temperature (K)
        
    c : 1x1 to 5x1 array 
        the coefficients for Equation 106, see documentation for `eq106`

    Returns
    -------
    float
        Value of temperature derivative of DIPPR Equation 106 at `tr` 
        given `tc` and `c`. 
        Units: default DIPPR units for property described by `c`
        divided by K. For HVP, J/(kmol*K)
    """
    a=fillcoeff(c)
    x = eq106(tr,a)/tc*((np.log(1-tr)*(a[2] + 2*a[3]*tr + 3*a[4]*tr**2)) - \
        (a[1] + a[2]*tr + a[3]*tr**2 + a[4]*tr**3)/(1-tr))
    return(x)  

def eq107(t,c):
    """DIPPR Equation 107
    
    Y = c[0] + c[1]*((c[2]/t)/np.sinh(c[2]/t))**2 + 
        c[3]*((c[4]/t)/np.cosh(c[4]/t))**2
    
    Property: ICP (J/(kmol*K))
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 5x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : see equation above
        c[2] : see equation above
        c[3] : see equation above
        c[4] : see equation above       
        
    Returns
    -------
    float
        Value of DIPPR Equation 107 at `t` given `c`. 
        Units: default DIPPR units for property described by `c` which
               should be J/(kmol*K)
    """
    if len(c) < 5:
        print("Error: DIPPR Equation 107 requires exactly 5 parameters.")
        return(float("NaN"))
    x = c[0] + c[1]*((c[2]/t)/np.sinh(c[2]/t))**2 + \
        c[3]*((c[4]/t)/np.cosh(c[4]/t))**2
    return(x)    
  
def eq114(tau,c):
    """DIPPR Equation 114
    
    Y = c[0]**2/tau + c[1] - 2*c[0]*c[2]*tau - c[0]*c[3]*tau**2 -  
        1/3*c[2]**2*tau**3 - 0.5*c[2]*c[3]*tau**4 - 1/5*c[3]**2*tau**5
    
    Property: LCP (J/(kmol*K)); limited use
    
    Parameters
    ----------
    tau : float
        one minus the reduced temperature, 1-tr (unitless)
        tr = 1 - t/tc
        
    c : 4x1 array 
        the coefficients for the equation
        c[0] : coefficient needed for 1/tau and tau terms
        c[1] : constant
        c[2] : coefficient needed for  tau, tau**3, and tau**4 terms
        c[3] : coefficient needed for tau**4 and tau**5 terms   
        
    Returns
    -------
    float
        Value of DIPPR Equation 114 at `tau` given `c`. 
        Units: default DIPPR units for property described by `c` which
               should be J/(mol*K)
    """
    if len(c) < 4:
        print("Error: DIPPR Equation 114 requires exactly 4 parameters.")
        return(float("NaN"))
    x = c[0]**2/tau + c[1] - 2*c[0]*c[2]*tau - c[0]*c[3]*tau**2 - \
        1/3*c[2]**2*tau**3 - 0.5*c[2]*c[3]*tau**4 - 1/5*c[3]**2*tau**5
    return(x) 
  
def eq115(t,c):
    """DIPPR Equation 115
    
    Y = np.exp(c[0] + c[1]/t + c[2]*np.log(t) + c[3]*t**2 + c[4]/t**2)
    
    Property: VP (Pa), SVP (Pa)
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 5x1 array 
        the coefficients for the equation
        c[0] : constant in exponent
        c[1] : coefficient on 1/t term in exponent
        c[2] : coefficient on ln(t) term in exponent
        c[3] : coefficient on t**2 term in exponent
        c[4] : coefficient on 1/t**2 term in exponent
        
    Returns
    -------
    float
        Value of DIPPR Equation 115 at `t` given `c`. 
        Units: default DIPPR units for property described by `c` which
               should be Pa
    """
    a=fillcoeff(c)
    x = np.exp(a[0] + a[1]/t + a[2]*np.log(t) + a[3]*t**2 + a[4]/t**2)
    return(x)

def eq116(tau,c):
    """DIPPR Equation 116
    
    Y = c[0] + c[1] * tau**0.35 + c[2]*tau**(2/3) + c[3]*tau + 
        c[4]*tau**(4/3)
    
    Property: LDN of water for a certain temperature range 
              (kmol/m**3)
    
    Parameters
    ----------
    tau : float
        one minus the reduced temperature, 1-tr (unitless)
        tr = 1 - t/tc
        
    c : 5x1 array 
        the coefficients for the equation
        c[0] : constant in exponent
        c[1] : coefficient on tau**0.35 term
        c[2] : coefficient on tau**(2/3) term
        c[3] : coefficient on tau term
        c[4] : coefficient on tau**(4/3) term
        
    Returns
    -------
    float
        Value of DIPPR Equation 116 at `tau` given `c`. 
        Units: default DIPPR units for property described by `c` which 
               should be kmol/m**3
    """
    if len(c) < 5:
        print("Error: DIPPR Equation 116 requires exactly 5 parameters.")
        return(float("NaN"))
    x = c[0] + c[1] * tau**0.35 + c[2]*tau**(2/3) + c[3]*tau + \
        c[4]*tau**(4/3)
    return(x)

def eq119(tau,c):
    """DIPPR Equation 119
    
    Y = c[0] + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau**(5/3) + 
        c[4]*tau**(16/3) + c[5]*tau**(43/3) + c[6]*tau**(110/3)
    
    Property: LDN of water for a certain temperature range 
              (kmol/m**3)
    
    Parameters
    ----------
    tau : float
        one minus the reduced temperature, 1-tr (unitless)
        tr = 1 - t/tc
        
    c : 7x1 array 
        the coefficients for the equation
        c[0] : constant in exponent
        c[1] : coefficient on tau**(1/3) term
        c[2] : coefficient on tau**(2/3) term
        c[3] : coefficient on tau**(5/3) term
        c[4] : coefficient on tau**(16/3) term
        c[5] : coefficient on tau**(43/3) term
        c[6] : coefficient on tau**(110/3) term
        
    Returns
    -------
    float
        Value of DIPPR Equation 119 at `tau` given `c`. 
        Units: default DIPPR units for property described by `c` which 
               should be kmol/m**3
    """ 
    if len(c) < 7:
        print("Error: DIPPR Equation 119 requires exactly 7 parameters.")
        return(float("NaN"))
    x = c[0] + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau**(5/3) + \
        c[4]*tau**(16/3) + c[5]*tau**(43/3) + c[6]*tau**(110/3)
    return(x)
  
def eq123(tau,c):
    """DIPPR Equation 123
    
    Jameson equation
    Y = c[0]*(1 + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau)
    
    Property: LTC (W/(m*K))
    
    Parameters
    ----------
    tau : float
        one minus the reduced temperature, 1-tr (unitless)
        tr = 1 - t/tc
        
    c : 4x1 array 
        the coefficients for the equation
        c[0] : base of power
        c[1] : coefficient on tau**(1/3) term in exponent
        c[2] : coefficient on tau**(2/3) term in exponent
        c[3] : coefficient on tau term in exponent
        
    Returns
    -------
    float
        Value of DIPPR Equation 123 at `tau` given `c`. 
        Units: default DIPPR units for property described by `c` which 
               should be W/(m*K)
    """
    if np.count_nonzero(c) != 4:
        print("Error: DIPPR Equation 123 requires exactly 4 parameters.")
        return(float("NaN"))
    x = c[0]*(1 + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau)
    return(x) 

def eq124(tau,c):
    """DIPPR Equation 124
    
    Y = c[0] + c[1]/tau + c[2]*tau + c[3]*tau**2 + c[4]*tau**3
    
    Property: LCP (J/(kmol*K))
    
    Parameters
    ----------
    tau : float
        one minus the reduced temperature, 1-tr (unitless)
        tr = 1 - t/tc
        
    c : 5x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : coefficient on 1/tau term
        c[2] : coefficient on tau term
        c[3] : coefficient on tau**2 term
        c[4] : coefficient on tau**3 term
        
    Returns
    -------
    float
        Value of DIPPR Equation 124 at `tau` given `c`. 
        Units: default DIPPR units for property described by `c` which 
               should be J/(kmol*K)
    """
    a=fillcoeff(c)
    x = a[0] + a[1]/tau + a[2]*tau + a[3]*tau**2 + a[4]*tau**3
    return(x)
  
def eq127(t,c):
    """DIPPR Equation 127
    
    Y = c[0] + c[1]*((c[2]/t)**2*np.exp(c[2]/t)/(np.exp(c[2]/t)-1)**2) + 
        c[3]*((c[4]/t)**2*np.exp(c[4]/t)/(np.exp(c[4]/t)-1)**2) + 
        c[5]*((c[6]/t)**2*np.exp(c[6]/t)/(np.exp(c[6]/t)-1)**2)
    
    Property: ICP (J/(kmol*K))
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 7x1 array 
        the coefficients for the equation
        c[0] : constant
        c[1] : see equation above
        c[2] : see equation above
        c[3] : see equation above
        c[4] : see equation above
        c[5] : see equation above
        c[6] : see equation above
        
    Returns
    -------
    float
        Value of DIPPR Equation 127 at `t` given `c`. 
        Units: default DIPPR units for property described by `c` which 
               should be J/(kmol*K)
    """
    if len(c) < 7:
        print("Error: DIPPR Equation 127 requires exactly 7 parameters.")
        return(float("NaN"))
    x = c[0] + c[1]*((c[2]/t)**2*np.exp(c[2]/t)/(np.exp(c[2]/t)-1)**2) + \
        c[3]*((c[4]/t)**2*np.exp(c[4]/t)/(np.exp(c[4]/t)-1)**2) + \
        c[5]*((c[6]/t)**2*np.exp(c[6]/t)/(np.exp(c[6]/t)-1)**2) 
    return(x)  
    
def eq(t,c,eqid):
    """Returns the value of DIPPR equation `eqid`
    
    This function returns the value of the DIPPR equation identified by
    `eqid` at `t` using the coefficients in `c`. 
    
    Parameters
    ----------
    t : float
        tempearture (K)
        
    c : 1x1 to 7x1 array 
        the coefficients for the equation identifed by `eqid`
        c[0] : first coefficient
        c[1] : second coefficient
        c[2] : third coefficient
        c[3] : fourth coefficient
        c[4] : fifth coefficient
        c[5] : sixth coefficient
        c[6] : seventh coefficient
    
    eqid : string
           Equation id for the desired correlation. May be 100, 101,
           102, 104, 105, 106, 107, 114, 115, 116, 119, 123,
           124, or 127.
        
        
    Returns
    -------
    float
        Value of DIPPR Equation `eqid at `t` given `c`. 
        Units: default DIPPR units for property described by `c` which 
    """
    if eqid == 100: return(eq100(t,c))
    if eqid == 101: return(eq101(t,c))
    if eqid == 102: return(eq102(t,c))
    if eqid == 104: return(eq104(t,c))
    if eqid == 105: return(eq105(t,c))
    if eqid == 106: return(eq106(t,c))
    if eqid == 107: return(eq107(t,c))
    if eqid == 114: return(eq114(t,c))
    if eqid == 115: return(eq115(t,c))
    if eqid == 116: return(eq116(t,c))
    if eqid == 119: return(eq119(t,c))
    if eqid == 123: return(eq123(t,c))
    if eqid == 124: return(eq124(t,c))
    if eqid == 127: return(eq127(t,c))
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
