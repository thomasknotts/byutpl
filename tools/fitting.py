# Copyright (C) 2023 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, fitting.py, is a python submodule of the dippr module. It     #
# that has function(s) that implement non-group contribution prediction    #
# methods.                                                                 #
#                                                                          #
# fitting.py is free software: you can redistribute it and/or              #
# modify it under the terms of the GNU General Public License as           #
# published by the Free Software Foundation, either version 3 of the       #
# License, or (at your option) any later version.                          #
#                                                                          #
# fitting.py is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with fitting.py.  If not, see                                      #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #

# ======================================================================== #
# fitting.py                                                               #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - April 2023                 Extended Trouton VP Correlation #
# ======================================================================== #

# ======================================================================== #
# fitting.py                                                               #
#                                                                          #
# This submodule contains function(s) to fit data to various correlations. #
# It requires other submodules in the dippr module.                        #
#                                                                          #
# The library can be loaded into python via the following command:         #
# import dippr.tools.fitting as fit                                        #
#                                                                          #
# -----------------------------------------------------------------------  #
# DEFINITION OF INPUT PARAMETERS FOR FUNCTIONS                             #
# -----------------------------------------------------------------------  #
# Symbol        Property                                    Units          #
# -----------------------------------------------------------------------  #
# t             array of temperatures                       K              #
# vp            array of vapor pressures at `t`             Pa             #
# nbp           normal boiling point                        K              #
# tau1          boolean delineating two vs one parameters   unitless       #
# -----------------------------------------------------------------------  #
#                                                                          #
#                                                                          #
# -------------------------------------------------------------------------------------------------  #
# Functions                                    Return Value                                Units     #
# -------------------------------------------------------------------------------------------------- #  
# troutonvp(t,vp,nbp,tau1=False,percent=False) tau0 and tau1 (see below)                   kJ/mol    #  
# ================================================================================================== #
     
import byutpl.tools.predictions as pred
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import numpy as np

def troutonvp(t, vp, nbp, tau1=False, percent=False):
    """fit the tau0 and tau1 Trouton VP parameters to data 
	
    Returns the parameter(s) in the extended Trouton
    Vapor Pressure Correlation [1] by fitting to the supplied data 
    for vapor pressure vs temperature. If tau0 is False, only tau0
    if used in the fitting. If tau1 is True, both tau0 and tau1 
    are fit. 
	
    Parameters
    ----------
    t : ndarray
        1D array of floats
        temperature data (K) at which vapor pressure `p` is known
    
    p : ndarray
        1D array of floats
        pressure data (Pa) for temperatures in `t`
        
    nbp : float
        the normal boiling point (K)
        
    tau1 : boolean, default=False
        True:  both tau0 and tau1 are fit to the data.
        False: only tau0 is fit to the data; tau1=0
    
    percent : boolean, default=False
        True: use the sum square percent difference as the objective 
              function when fitting
        False: use the sum square error as the objective function 
               when fitting
        
    Returns
    -------
    array of floats
        The values of the parameters of the extended Trouton Vapor
        Pressure correlation in kJ/mol

    References
    ----------
    .. [1] P. M Mathias, G. Jacobs, J. Cabrera, Modified Trouton’s Rule for the 
       Estimation, Correlation, and Evaluation of Pure-Component Vapor Pressure, 
       J. Chem. Eng. Data, 63, 943-953 (2018).
	"""
    if percent==False:
        if tau1==False:
            def f(t,tau0guess):
                return(pred.troutonvp(t,nbp,tau0=tau0guess))
        else:
            def f(t,tau0guess, tau1guess):
                return(pred.troutonvp(t,nbp,tau0=tau0guess,tau1=tau1guess))
        param,cov,info,msg,soln=curve_fit(f,t,vp,full_output=True)
        return(param)
    else:
        if tau1==False:
            def f(tau0guess,t):
                resid=(pred.troutonvp(t,nbp,tau0=tau0guess)/vp-1)*100
                return(np.sum(resid*resid))
            tauguess=np.array([1])
        else:
            def f(tausguesses,t):
                tau0guess, tau1guess = tausguesses
                resid=(pred.troutonvp(t,nbp,tau0=tau0guess,tau1=tau1guess)/vp-1)*100
                return(np.sum(resid*resid)) 
            tauguess=np.array([1,1])
        param=minimize(f,tauguess,args=(t))
        return(param.x)
        
        
    
