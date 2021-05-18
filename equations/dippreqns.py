# Copyright (C) 2019 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, dippreqns.py, is a python module with functions that          #
# accept a temperature in K and an array of coefficients and returns the   #
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
# ======================================================================== #
"""
This module contains functions that accept a temperature in K and an
array of coefficients and returns the  value of the DIPPR equation.      

The equations are numbered as listed in the DIPPR Policies and 
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
tr                    reduced system temperature         unitless       
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
		
	c : 5x1 array 
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
    x = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4
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
		
	c : 5x1 array 
		the coefficients for the equation
		c[0] : constant
		c[1] : coefficient on 1/t term
		c[2] : coefficient on ln(t) term
		c[3] : coefficient on t**c[4] term
		c[4] : power on `t` for c[3] term
	
	Returns
	-------
	float
		Value of DIPPR Equation 101 at `t` given `c`. 
		Units: default DIPPR units for property described by `c`.
	"""  
	x = np.exp(c[0] + c[1]/t + c[2]*np.log(t) + c[3]*t**c[4])
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
		
	c : 5x1 array 
		the coefficients for Equation 101; see documentation for `eq101`
	
	Returns
	-------
	float
		Value of temperature derivative of the natural logarithm
		of DIPPR Equation 101 at `t` given `c`. 
		Units: default DIPPR units for property described by `c`
		divided by K. For VP or SVP, Pa/K
	"""
    x = -1.0*c[1]/t**2 + c[2]/t + c[3]*c[4]*t**(c[4]-1.0)
    return(x)

def eq102(t,c):
	"""DIPPR Equation 102
	
	Y = c[0]*t**c[1]/(1 + c[2]/t + c[3]/t**2)
    
    Properties: SCP (J/(kmol*K)), VVS (Pa*s), VTC (W/(m*K))
	
	Parameters
	----------
	t : float
		tempearture (K)
		
	c : 4x1 array 
		the coefficients for the equation
		c[0] : coefficient on t**c[1] term
		c[1] : power on `t` for c[0] term
		c[2] : coefficient on 1/t term
		c[3] : coefficient on 1/t**2 term
	
	Returns
	-------
	float
		Value of DIPPR Equation 102 at `t` given `c`. 
		Units: default DIPPR units for property described by `c`.
	"""  
    x = c[0]*t**c[1]/(1 + c[2]/t + c[3]/t**2)
    return(x)
  
def eq104(t,c):
	"""DIPPR Equation 104
	
	Y = c[0] + c[1]/t + c[2]/t**3 + c[3]/t**8 + c[4]/t**9
    
    Property: SVR (m**3/kmol)
	
	Parameters
	----------
	t : float
		tempearture (K)
		
	c : 5x1 array 
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
		Units: default DIPPR units for property described by `c`.
	"""
    x = c[0] + c[1]/t + c[2]/t**3 + c[3]/t**8 + c[4]/t**9
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
		
	c : 4x1 array 
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
    x = eq105(t,c)*c[3]/c[2]*np.log(c[1])*(1 - t/c[2])**(c[3]-1)
    return(x)

def eq106(tr,c):
	"""DIPPR Equation 106
	
	Y = c[0]*(1-tr)**(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)
    
    Property: HVP (J/kmol), ST (N/m)
	
	Parameters
	----------
	t : float
		tempearture (K)
		
	c : 5x1 array 
		the coefficients for the equation
		c[0] : base of power term
		c[1] : constant in exponent
        c[2] : coefficient on tr term in exponent
        c[3] : coefficient on tr**2 term in exponent
        c[4] : coefficient on tr**3 term in exponent       
        
	Returns
	-------
	float
		Value of DIPPR Equation 106 at `t` given `c`. 
		Units: default DIPPR units for property described by `c`.
	"""
    x = c[0]*(1-tr)**(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)
    return(x)

def eq106a(tr,tc,c):
	"""DIPPR Equation 106a
	
	The temperature derivative of Equation 106. This is not an official
    DIPPR equation. It is usually used to obtain dHVP/dt.
	
	Parameters
	----------
	t : float
		tempearture (K)
		
	c : 5x1 array 
		the coefficients for Equation 106, see documentation for `eq106`

	Returns
	-------
	float
		Value of temperature derivative of DIPPR Equation 106 at `t` 
        given `c`. 
		Units: default DIPPR units for property described by `c`
		divided by K. For HVP, J/(kmol*K)
	""" 
    x = eq106(tr,c)/tc*((np.log(1-tr)*(c[2] + 2*c[3]*tr + 3*c[4]*tr**2))-(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)/(1-tr))
    return(x)  

def eq107(t,c):
	"""DIPPR Equation 107
	
	Y = c[0] + c[1]*((c[2]/t)/np.sinh(c[2]/t))**2 + \
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
		Units: default DIPPR units for property described by `c`.
	"""
    x = c[0] + c[1]*((c[2]/t)/np.sinh(c[2]/t))**2 + c[3]*((c[4]/t)/np.cosh(c[4]/t))**2
    return(x)    
  
def eq114(tau,c): # DIPPR Equation 114
  x = c[0]**2/tau + c[1] - 2*c[0]*c[2]*tau - c[0]*c[3]*tau**2 - 1/3*c[2]**2*tau**3 - 0.5*c[2]*c[3]*tau**4 - 1/5*c[3]**2*tau**5
  return(x) # DIPPR default units for coefficients supplied    
  
def eq115(t,c): # DIPPR Equation 115
  x = np.exp(c[0] + c[1]/t + c[2]*np.log(t) + c[3]*t**2 + c[4]/t**2)
  return(x) # DIPPR default units for coefficients supplied      

def eq116(tau,c): # DIPPR Equation 116
  x = c[0] + c[1] * tau**0.35 + c[2]*tau**(2/3) + c[3]*tau + c[4]*tau**(4/3)
  return(x) # DIPPR default units for coefficients supplied

def eq119(tau,c): # DIPPR Equation 119
  x = c[0] + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau**(5/3) + c[4]*tau**(16/3) + c[5]*tau**(43/3) + c[6]*tau**(110/3)
  return(x) # DIPPR default units for coefficients supplied  
  
def eq123(tau,c): # DIPPR Equation 123
  x = c[0]*(1 + c[1]*tau**(1/3) + c[2]*tau**(2/3) + c[3]*tau)
  return(x) # DIPPR default units for coefficients supplied    

def eq124(tau,c): # DIPPR Equation 124
  x = c[0] + c[1]/tau + c[2]*tau + c[3]*tau**2 + c[4]*tau**3
  return(x) # DIPPR default units for coefficients supplied
  
def eq127(t,c): # DIPPR Equation 127
  x = c[0] + c[1]*((c[2]/t)**2*np.exp(c[2]/t)/(np.exp(c[2]/t)-1)**2) + c[3]*((c[4]/t)**2*np.exp(c[4]/t)/(np.exp(c[4]/t)-1)**2) + c[5]*((c[6]/t)**2*np.exp(c[6]/t)/(np.exp(c[6]/t)-1)**2) 
  return(x) # DIPPR default units for coefficients supplied     
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
