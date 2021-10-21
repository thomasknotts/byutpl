# Copyright (C) 2019 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, predictions.py, is a python submodule of the dippr module. It #
# that has function(s) that implement non-group contribution prediction    #
# methods.                                                                 #
#                                                                          #
# predictions.py is free software: you can redistribute it and/or          #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# predictions.py is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with predictions.py.  If not, see                                  #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# predictions.py                                                           #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - October 2019             Derivative Method for LCP         #
# ======================================================================== #

# ======================================================================== #
# predictions.py                                                           #
#                                                                          #
# This submodule contains function(s) that perform non-group contributions #
# based predictions. It requires other submodules in the dippr module.     #
#                                                                          #
# The library can be loaded into python via the following command:         #
# import dippr.predictions as pred                                         #
#                                                                          #
# -----------------------------------------------------------------------  #
# DEFINITION OF INPUT PARAMETERS FOR FUNCTIONS                             #
# -----------------------------------------------------------------------  #
# Symbol        Property                                    Units          #
# -----------------------------------------------------------------------  #
# t             system temperature                          K              #
# psat			vapor pressure at temp t                    Pa             #
# tr            reduced system temperature                  unitless       #
# tc            critical temperature                        K              #
# pc            critical pressure                           Pa             #
# w             acentric factor                             unitless       #
# eqnicp		dippr equation number for ICP correlation   unitless       #
# cicp          matrix of coefficients for ICP              DIPPR Default  #
# cvp           matrix of coefficients for icp              DIPPR Default  #
# chvp          matrix of coefficients for icp              DIPPR Default  #
# cldn          matrix of coefficients for icp              DIPPR Default  #
# -----------------------------------------------------------------------  #
#                                                                          #
#                                                                          #
# -------------------------------------------------------------------------------------------------  #
# AVAILABLE FUNCTIONS                                                                                # 
# -------------------------------------------------------------------------------------------------  #
# Auxiliary Function                           Return Value                              Units       #
# -------------------------------------------------------------------------------------------------- #  
# dVPdT(t,cvp)                                 temperature derivative of vapor pressure  Pa/K        #
#                                              (used by sigmaToPCorrectionV)                         #
# dHVPdT(tr,tc,chvp)                           temperature derivative of heat of vap.    J/kmol/K    #
#                                              (used by LCPder)
# drLDNdT(t,cldn)                              temperature derivative of the reciprocal  m**3/kmol/K #
#                                              of liquid density                                     #
#                                              (used by sigmaToPCorrectionL)                         #
# ICP(t,cicp,eqnicp)                           ideal gas heat capacity at t              J/kmol/K    #
#                                              (used by LCPder)                                      #
# IdealToRealGasCorrection(t,psat,tc,pc,w)     correction from ideal to real gas         J/kmol/K    #
#                                              heat capacity (used by LCPder)                        #
# sigmaToPCorrectionV(t,psat,tc,pc,w,cvp)      correction from saturation to constant P  J/kmol/K    #
#                                              heat capacity of the vapor at t                       #
#                                              (used by LCPder)                                      #
# sigmaToPCorrectionL((t,tc,pc,w,cvp,cldn)     correction from saturation to constant P  J/kmol/K    #
#                                              heat capacity of the vapor at t                       #
#                                              (used by LCPder)                                      #

# -------------------------------------------------------------------------------------------------  #
# Primary Function                           Return Value                                Units       #
# -------------------------------------------------------------------------------------------------- #  
# LCPder(t,tc,pc,w,cicp,eqnicp,cvp,chvp,cldn)  liquid heat capacity at temperature t     J/kmol/K    #
#                                              predicted using the derivative method                 # 
# ================================================================================================== #

import numpy as np
import byutpl.eos.srk as srk
import byutpl.equations.dippreqns as dippr          

def ICP(t,cicp,eqnicp):
    """ideal gas heat capacity

    Returns the ideal gas isobaric heat capacity from the 
    DIPPR(R) [1] correlation
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    cicp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the ideal gas isobaric heat capacity correlation of the compound
    
    eqnicp : integer
        the DIPPR(R) equation number corresponding to the `cicp`
        
    Returns
    -------
    float
        the value of the ideal gas isobaric heat capacity at `t` in J kmol**-1 K**-1
        
    References
    ----------    
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    if (eqnicp == "127" or eqnicp == 127):
        x = dippr.eq127(t,cicp)
    else:
        x = dippr.eq107(t,cicp)
    return(x)
    
def IdealToRealGasCpCorrection(t,p,tc,pc,w):
    """correction from ideal to real gas vapor heat capacity
	
    Returns the change in vapor heat capacity when moving from ideal to real
    constant pressure heat capacity of the vapor. The correction is calculated
    using the SRK Equation of state.
    
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    p : float
        The pressure (Pa) of the system
    
    tc : float
        The critical temperature (K) of the compound
    
    pc : float
        The critical pressure (Pa) of the compound
        
    w : float
        The acentric factor (unitless) of the compound
           
    Returns
    -------
    float
        The change in vapor isobaric heat capacity when moving from the ideal to 
        the real state in J kmol**-1 K**-1

	"""
    v = srk.vv(t,p,tc,pc,w)
    b = srk.b(tc,pc)
    x = t*srk.d2ThetadT2(t,tc,pc,w)*np.log(v/(v+b))/b + t*srk.dVdT(t,v,tc,pc,w)**2*srk.dPdV(t,v,tc,pc,w)+srk.rg
    x = x*1000 # convert from J/mol/K to J/kmol/K
    return(x)

def dVPdT(t,cvp):
    """temperature derivative of vapor pressure
	
    Returns the temperature derivative of the vapor pressure
    evaluated at `t`. Uses the DIPPR(R) [1] vapor pressure correlation.
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    cvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the vapor pressure correlation of the compound
        
    Returns
    -------
    float
        the temperature derivative of the vapor pressure
        evaluated at `t` in units of Pa/K
        
    References
    ----------    
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(dippr.eq101(t,cvp)*dippr.eq101a(t,cvp))



def sigmaToPCorrectionV(t,tc,pc,w,cvp):
    """correction from saturation to contant P vapor heat capacity
	
    Returns the change in vapor heat capacity when moving from saturation
    (C_sigma_V) to constant pressure (C_p_V). This value is added
    to C_sigma_V to obtain  C_p_V. The correction is calculated using the 
    SRK Equation of state with DIPPR (R) [1] correlations for properties
    as explained below.
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    tc : float
        The critical temperature (K) of the compound
    
    pc : float
        The critical pressure (Pa) of the compound
        
    w : float
        The acentric factor (unitless) of the compound
    
    cvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the vapor pressure correlation of the compound
        
    Returns
    -------
    float
        The change in vapor heat capacity when moving from saturation
        (C_sigma_V) to constant pressure (C_p_V) in J kmol**-1 K**-1

    References
    ----------      
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    psat = dippr.eq101(t,cvp)
    v = srk.vv(t,psat,tc,pc,w)
    x = (v - t*srk.dVdT(t,v,tc,pc,w))*dVPdT(t,cvp)
    x = x*1000 # convert from J/mol/K to J/kmol/K
    return(x)
	
def dHVPdT(tr,tc,chvp):
    """temperature derivative of heat of vaporization
	
    Returns the temperature derivative of the heat of vaporization
    evaluated at `tr`. Uses the DIPPR(R) [1] heat of vaporization correlation.
	
    Parameters
    ----------
    tr : float
        The temperature (K) of the system
        
    chvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the heat of vaporization correlation of the compound
        
    Returns
    -------
    float
        the temperature derivative of the reciprocal of the heat of vaporization
        evaluated at `tr` in units of J kmol**-1 K**-1

    References
    ----------    
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(dippr.eq106a(tr,tc,chvp))
	
def drLDNdT(t,cldn): # temperature derivative of the reciprocal of liquid density
    """temperature derivative of the reciprocal of liquid density
	
    Returns the temperature derivative of the reciprocal of liquid density
    evaluated at `t`. Uses the DIPPR(R) [1] liquid density correlation.
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    cldn : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the liquid density correlation of the compound
        
    Returns
    -------
    float
        the temperature derivative of the reciprocal of liquid density
        evaluated at `t` in units of m**3 kmol**-1 K**-1

    References
    ----------    
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(-1.0/dippr.eq105(t,cldn)*cldn[3]/cldn[2]*np.log(cldn[1])* \
           (1-t/cldn[2])**(cldn[3]-1))


def sigmaToPCorrectionL(t,tc,pc,w,cvp,cldn):
    """correction from saturation to contant P liquid heat capacity
	
    Returns the change in liquid heat capacity when moving from 
    saturation (C_sat_L) to constant pressure (C_p_L). This value is added
    to C_sat_L to obtain  C_p_L. The correction is calculated using the 
    BHT correlation [1] with DIPPR (R) [2] correlations for properties
    as explained below.
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    tc : float
        The critical temperature (K) of the compound
    
    pc : float
        The critical pressure (Pa) of the compound
        
    w : float
        The acentric factor (unitless) of the compound
    
    cvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the vapor pressure correlation of the compound
        
    cldn : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the liquid density correlation of the compound
        
    Returns
    -------
    float
        The value of the change in liquid heat capacity when moving from 
        saturation (C_sat_L) to constant pressure (C_p_L) in J kmol**-1 K**-1

    References
    ----------
    .. [1] G. H. Thomson and K. R. Brobst and R. W. Hankinson, An improved 
       correlation for densities of compressed liquids and liquid mixtures,
       AIChE Journal, 28:4, 671-676 (1982).
       
    .. [2] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    # Coefficients for BHT method
    a=-9.070217
    b=62.45326
    j=0.0861488
    k=0.0344483
    c=j+w*k
    d=-135.1102
    f=4.79594
    g=0.250047
    h=1.14188
    e=np.exp(f+g*w+h*w**2)
    
    # Properties and intermediate calculations
    tr = t/tc # reduced temperature
    tau = 1-tr 
    ldn = dippr.eq105(t,cldn) # liquid density converted in kmol/m**3
    drldndt = drLDNdT(t,cldn) # temperature derivative of the reciprocal of liquid density in kmol/m**3/K
    vp = dippr.eq101(t,cvp) # vapor pressure in Pa
    dvpdt = dVPdT(t,cvp) # temperature derivative of vapor pressure in Pa/K
    beta = pc*(-1.0 + a*tau**(1/3) + b*tau**(2/3) + d*tau + e*tau**(4/3)) # Pa
    
    # the value of the function in J/kmol/K 
    x = dvpdt*(1/ldn - t*(drldndt+c/ldn/(beta+vp)*dvpdt))
    return(x)

def LCPder(t,tc,pc,w,cicp,eqnicp,cvp,chvp,cldn):
    """liquid heat capacity prediction from DIPPR(R) derivative method
	
    Liquid heat capacity predicted from the DIPPR(R) [1] derivative method.
    This technique is a thermodynamically rigorous method to use  
    correlations for the ideal gas heat capacity, the vapor pressure,
    the heat of vaporization, and the liquid density of a compound
    to predict the liquid heat capacity.
	
    Parameters
    ----------
    t : float
        The temperature (K) of the system
        
    tc : float
        The critical temperature (K) of the compound
    
    pc : float
        The critical pressure (Pa) of the compound
        
    w : float
        The acentric factor (unitless) of the compound
    
    cicp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the ideal gas heat capacity correlation of the compound
    
    eqnicp : integer
        DIPPR(R) equation number for the ideal gas heat capacity 
        correlation of the compound
    
    cvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the vapor pressure correlation of the compound

    chvp : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the heat of vaporization correlation of the compound
        
    cldn : ndarray
        1D numpy array containing the DIPPR(R) coefficients (A, B, C, ...)
        for the liquid density correlation of the compound
        
    Returns
    -------
    float
        The value of the liquid heat capcity (J kmol**-1 K**-1) predicted
        using the DIPPR(R) derivative method

    References
    ----------
    .. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    psat = dippr.eq101(t,cvp)
    return(ICP(t,cicp,eqnicp) - dHVPdT(t/tc,tc,chvp) - \
           IdealToRealGasCpCorrection(t,psat,tc,pc,w) + \
           sigmaToPCorrectionV(t,tc,pc,w,cvp) - \
           sigmaToPCorrectionL(t,tc,pc,w,cvp,cldn))

