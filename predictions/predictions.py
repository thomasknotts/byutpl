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
import dippr.equations as dippr # DIPPR equations by number            
import dippr.srkprops as srk # PVT properties from SRK EOS  

# ------------------------------------------------------------------------ #
# Functions                                                                #
# ------------------------------------------------------------------------ #

# ICP: Ideal Gas Heat Capacity

# Function Input Parameters
# * t: temperature in K
# * c: coefficient matrix for the ICP correlation
# * eqnicp: the equation number for the correlation

# Function Return Value
# the ideal gas heat capacity at t in J kmol**-1 K**-1
def ICP(t,cicp,eqnicp): 
    if (eqnicp == "127" or eqnicp == 127):
        x = dippr.eq127(t,cicp)
    else:
        x = dippr.eq107(t,cicp)
    return(x) # J/kmol/K


# IdealToRealGasCorrection

# Function Input Parameters
# * t: temperature in K
# * psat: vapor pressure in Pa at temperature t
# * tc: critical temperature in K
# * pc: critical pressure in K
# * w: acentric factor

# Function Return Value
# the correction from ideal to real gas heat capacity at t in units of J/kmol/K
def IdealToRealGasCorrection(t,psat,tc,pc,w):
    v = srk.vv(t,psat,tc,pc,w)
    b = srk.b(tc,pc)
    x = t*srk.d2ThetadT2(t,tc,pc,w)*np.log(v/(v+b))/b + t*srk.dVdT(t,v,tc,pc,w)**2*srk.dPdV(t,v,tc,pc,w)+srk.rg
    x = x*1000 # convert from J/mol/K to J/kmol/K
    return(x) # J/kmol/K


# dVPdT

# Function Input Parameters
# * t: temperature in K
# * cvp: matrix with coefficients for VP correlation

# Function Return Value
# the temperature derivative of the vapor pressure correlation with respect to temperature in units of Pa/K
def dVPdT(t,cvp):
    return(dippr.eq101(t,cvp)*dippr.eq101a(t,cvp)) # Pa/K


# sigmaToPCorrectionV

# Function Input Parameters
# * t: temperature in K
# * psat: vapor pressure in Pa at temperature t
# * tc: critical temperature in K
# * pc: critical pressure in K
# * w: acentric factor
# * cvp: matrix with coefficients for VP correlation

# Function Return Value
# the correction from saturation to constant P heat capacity of the vapor at t in units of J/kmol/K
def sigmaToPCorrectionV(t,psat,tc,pc,w,cvp):
    v = srk.vv(t,psat,tc,pc,w)
    x = (v - t*srk.dVdT(t,v,tc,pc,w))*dVPdT(t,cvp)
    x = x*1000 # convert from J/mol/K to J/kmol/K
    return(x) # J/kmol/K
	

#dHVPdT

# Function Input Parameters
# * tr: reduced temperature
# * tc: critical temperature in K
# * chvp: matrix with coeffcients for HVP correlation

# Function Return Value
# the derivative of HVP with respect to temperature t at tr in units of J/kmol/K
def dHVPdT(tr,tc,chvp):
	return(dippr.eq106a(tr,tc,chvp)) # J/kmol/K
	
	
#drLDNdT

# Function Input Parameters
# * t: temperature in K
# * cldn: matrix with coefficients for LDN correlation

# Function Return Value
# temperature derivative of the reciprocal of liquid density, evaluated at temperature t, returned in units of m**3/kmol/K
def drLDNdT(t,cldn): # temperature derivative of the reciprocal of liquid density
    return(-1.0/dippr.eq105(t,cldn)*cldn[3]/cldn[2]*np.log(cldn[1])*(1-t/cldn[2])**(cldn[3]-1)) # m**3/kmol/K


# sigmaToPCorrectionL

# Function Input Parameters
# * t: temperature in K
# * tc: critical temperature in K
# * pc: critical pressure in K
# * w: acentric factor
# * cvp: matrix with coefficients for VP correlation
# * cldn: matrix with coefficients for LDN correlation

# Function Return Value
# the correction from saturation to constant P heat capacity of the liquid at t in units of J/kmol/K
# calculated using the BHT correlation (G. H. Thomson and K. R. Brobst and R. W. Hankinson, An 
# improved correlation for densities of compressed liquids and liquid mixtures, AIChE Journal, 28:4, 671-676 (1982).)
def sigmaToPCorrectionL(t,tc,pc,w,cvp,cldn):
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
    return(x) # J/kmol/K


# LCPder

# Function Input Parameters
# * t: temperature in K
# * tc: critical temperature in K
# * pc: critical pressure in K
# * w: acentric factor
# * cicp: matrix with coefficients for ICP correlation
# * eqnicp: dippr equation number for ICP correlation
# * cvp: matrix with coefficients for VP correlation
# * cldn: matrix with coefficients for LDN correlation

# Function Return Value
# The liquid heat capacity at temperature t predicted using the derivative method; given in units of J/kmol/K

def LCPder(t,tc,pc,w,cicp,eqnicp,cvp,chvp,cldn):
    psat = dippr.eq101(t,cvp)
    return(ICP(t,cicp,eqnicp)-dHVPdT(t/tc,tc,chvp)-IdealToRealGasCorrection(t,psat,tc,pc,w)+sigmaToPCorrectionV(t,psat,tc,pc,w,cvp)-sigmaToPCorrectionL(t,tc,pc,w,cvp,cldn)) # J/kmol/K

