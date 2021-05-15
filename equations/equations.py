# Copyright (C) 2019 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, dippres.py, is a python library that has functions that     #
# accepts a temperature in K and an array of coefficients and returns the  #
# value of the DIPPR equation.                                             #
#                                                                          #
# dippres.py is free software: you can redistribute it and/or            #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# dippres.py is distributed in the hope that it will be useful,          #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with dippres.py.  If not, see                                    #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# dippres.py                                                             #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - September 2019                                             #
# ======================================================================== #

# ======================================================================== #
# dippres.py                                                             #
#                                                                          #
# This library contains functions that accepts a temperature in K and an   #
# array of coefficients and returns the  value of the DIPPR equation.      #
#                                                                          #
# The equations are listed in the DIPPR Policies and Procedures Manual (C) #
#                                                                          #
# The library can be loaded into python via the following command:         #
# import dippres as dippr                                                #
#                                                                          #
# ----------------------------------------------------------------------   #
# DEFINITION OF INPUT PARAMETERS FOR FUNCTIONS                             #
# ----------------------------------------------------------------------   #
# Symbol                Property                            Units          #
# ----------------------------------------------------------------------   #
# t                     system temperature                  K              #
# tr                    reduced system temperature          unitless       #
# tau                   1 - tr                              unitless       #
# c                     matrix of coefficients              DIPPR Default  #
# ----------------------------------------------------------------------   #
#                                                                          #
#                                                                          #
# ----------------------------------------------------------------------   #
# AVAILABLE PROPERTY FUNCTIONS                                             # 
# ----------------------------------------------------------------------   #
# Function              Return Value                        Units          #
# ----------------------------------------------------------------------   #
# eq100(t,c)            value of Eqn. 100 at t with c       DIPPR Default  #
# eq101(t,c)            value of Eqn. 101 at t with c       DIPPR Default  #
# eq102(t,c)            value of Eqn. 102 at t with c       DIPPR Default  #
# eq104(t,c)            value of Eqn. 104 at t with c       DIPPR Default  #
# eq105(t,c)            value of Eqn. 105 at t with c       DIPPR Default  #
# eq106(tr,c)           value of Eqn. 106 at tr with c      DIPPR Default  #
# eq107(t,c)            value of Eqn. 107 at t with c       DIPPR Default  #
# eq114(tau,c)          value of Eqn. 114 at tau with c     DIPPR Default  #
# eq115(t,c)            value of Eqn. 115 at t with c       DIPPR Default  #
# eq116(tau,c)          value of Eqn. 116 at tau with c     DIPPR Default  #
# eq119(tau,c)          value of Eqn. 119 at tau with c     DIPPR Default  #
# eq123(tau,c)          value of Eqn. 123 at tau with c     DIPPR Default  #
# eq124(tau,c)          value of Eqn. 124 at tau with c     DIPPR Default  #
# eq127(t,c)            value of Eqn. 127 at t with c       DIPPR Default  #
# ======================================================================== #

import numpy as np

# ------------------------------------------------------------------------ #
# DIPPR Equations                                                          #
# ------------------------------------------------------------------------ #

def eq100(t,c): # DIPPR Equation 100
    x = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4
    return(x) # DIPPR default units for coefficients supplied
    
def eq101(t,c): # DIPPR Equation 101
  x = np.exp(c[0] + c[1]/t + c[2]*np.log(t) + c[3]*t**c[4])
  return(x) # DIPPR default units for coefficients supplied 

def eq101a(t,c): # DIPPR Equation 101a
  x = -1.0*c[1]/t**2 + c[2]/t + c[3]*c[4]*t**(c[4]-1.0)
  return(x) # DIPPR default units for coefficients supplied 

def eq102(t,c): # DIPPR Equation 102
  x = c[0]*t**c[1]/(1 + c[2]/t + c[3]/t**2)
  return(x) # DIPPR default units for coefficients supplied 
  
def eq104(t,c): # DIPPR Equation 104
  x = c[0] + c[1]/t + c[2]/t**3 + c[3]/t**8 + c[4]/t**9
  return(x) # DIPPR default units for coefficients supplied 
 
def eq105(t,c): # DIPPR Equation 105
  x = c[0]/c[1]**(1 + (1 - t/c[2])**c[3])
  return(x) # DIPPR default units for coefficients supplied
  
def eq105a(t,c): # DIPPR Equation 105a (This is the temperature derivative of Equation 105. It is not an offical DIPPR equation.)
  x = eq105(t,c)*c[3]/c[2]*np.log(c[1])*(1 - t/c[2])**(c[3]-1)
  return(x) # DIPPR default units for coefficients supplied (for LDN likely kmol/m**3/K

def eq106(tr,c): # DIPPR Equation 106
  x = c[0]*(1-tr)**(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)
  return(x) # DIPPR default units for coefficients supplied   

def eq106a(tr,tc,c): # DIPPR Equation 106a (This is the temperature derivative of Equation 106. It is not an official DIPPR equation.)
  x = eq106(tr,c)/tc*((np.log(1-tr)*(c[2] + 2*c[3]*tr + 3*c[4]*tr**2))-(c[1] + c[2]*tr + c[3]*tr**2 + c[4]*tr**3)/(1-tr))
  return(x) # DIPPR default units for coefficients supplied (for HVP likely J/kmol/K)   

def eq107(t,c): # DIPPR Equation 107
  x = c[0] + c[1]*((c[2]/t)/np.sinh(c[2]/t))**2 + c[3]*((c[4]/t)/np.cosh(c[4]/t))**2
  return(x) # DIPPR default units for coefficients supplied     
  
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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  