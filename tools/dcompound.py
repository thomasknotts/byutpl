# dcompound is used to store and retrieve DIPPR compound data               #
# It uses the 'unum' package to add units support.                          #
# Copyright (C) 2022 Thomas Allen Knotts IV - All Rights Reserved           #
#                                                                           #
# This program is free software you can redistribute it and/or modify       #
# it under the terms of the GNU General Public License as published by      #
# the Free Software Foundation, either version 3 of the License, or         #
# (at your option) any later version.                                       #
#                                                                           #
# This program is distributed in the hope that it will be useful,           #
# but WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
# GNU General Public License for more details.                              #
#                                                                           #
# You should have received a copy of the GNU General Public License         #
# along with this program.  If not, see httpwww.gnu.orglicenses.            #
# ========================================================================= #
# dcompound.py                                                              #
#                                                                           #
# Thomas A. Knotts IV                                                       #
# Brigham Young University                                                  #
# Department of Chemical Engineering                                        #
# Provo, UT  84606                                                          #
# Email thomas.knotts@byu.edu                                               #
# ========================================================================= #
# Version 1.0 - June 2023                                                   #
# ========================================================================= #
"""
This module containes the classes to store and retrieve the DIPPR data for 
a compound. It is a "dimensionalized" version of the 'compound' library
in this package.

======================================================================
Classes
======================================================================
compound       holds the property information for each compound
tcoeff         holds the information for a temperature-dependent
               property; not usually called directly by user
mdata          holds the meta data for constant properties

----------------------------------------------------------------------
Key Method of `compound` Class
----------------------------------------------------------------------
TO DO: read_compound(fn)
  reads the data into object `compound` from a file name `fn`
----------------------------------------------------------------------

======================================================================
General Use Instructions
======================================================================
1. initialize a `compound object
   c=compound()
2. Add data to the object
   a. Use c.read_compound('filename.txt')
   b. Manually assign the attributes

Once initialized, constant property `cprop` for `compound` object `c`
is accessed by 
  c.cprop
For example,
  c.TC
returns the critical temperature of compound object `c`.

Temperature dependent property `tprop` for compound object `c` 
at temperature `t` in kelvin is accessed by
  c.tprop(t)
For example,
  c.HVP(300)
returns the heat of vaporization of compond object `c` at 300 K.

======================================================================
Constant Properties
----------------------------------------------------------------------
Symbol          Description                                     Units
======================================================================
MW         molecular weight                                    kg/kmol
TC         critical temperature                                      K
PC         critical pressure                                        Pa
VC         critical volume                                   m**3/kmol
ZC         critical compressibility factor                    unitless
MP         melting point                                             K
TPT        triple point temperature                                  K
TPP        triple point pressure                                    Pa
NBP        normal boiling point                                      K
LVOL       liquid molar volume                               m**3/kmol
HFOR       ideal gas heat of formation                          J/kmol
GFOR       ideal gas Gibbs energy of formation                  J/kmol
ENT        ideal gas absolute entropy                       J/(kmol*K)
HSTD       standard state heat of formation                     J/kmol
GSTD       standard state Gibbs energy of formation             J/kmol
SSTD       standard state entropy of formation              J/(kmol*K)
HFUS       heat of fusion                                       J/kmol
HCOM       standard heat of combusion                           J/kmol
ACEN       acentric factor                                    unitless
RG         radius of gyration                                        m
SOLP       solubility parameter                          (J/m**3)**0.5
DM         dipole moment                                           C*m
VDWA       van der Waals Area                                     m**2
VDWV       van der Waals Volume                                   m**3
RI         refractive index                                   unitless
FP         flash point                                               K
FLVL       lower flammability limit                              % vol
FLTL       lower flammability limit temperature                      K
FLVU       upper flammability limit                              % vol
FLTU       upper flammability limit temperature                      K
AIT        autoignition temperature                                  K
HSUB       heat of sublimation                                  J/kmol
PAR        parachor                                           unitless
DC         dielectric constant                                unitless
======================================================================

======================================================================
Temperature-dependent Properties
----------------------------------------------------------------------
Symbol          Description                                     Units
======================================================================
LDN        saturated liquid density                          kmol/m**3
SDN        solid density                                     kmol/m**3
ICP        ideal gas heat capacity                          J/(kmol*K)
LCP        liquid heat capacity                             J/(kmol*K)
SCP        solid heat capacity                              J/(kmol*K)
HVP        heat of vaporization                                 J/kmol
SVR        second virial coefficient                         m**3/kmol
ST         surface tension                                         N*m
VTC        vapor thermal conductivity                          W/(m*K)
LTC        liquid thermal conductivity                         W/(m*K)
LTC        solid thermal conductivity                          W/(m*K)
VP         vapor pressure                                           Pa
SVP        solid vapor pressure                                     Pa
LVS        liquid viscosity                                       Pa*s
VVS        low pressure vapor viscosity                           Pa*s
======================================================================

"""
import sys, os, string, math
import numpy as np
from unum import Unum
import unum.units as u
import byutpl.equations.dippreqns as eq

# returns the units corresponding to 'prop'
def add_dippr_units(prop):
    temp_props=['MP','NBP','TC','TPT','AIT','FP','FLTL','FLTU']
    press_props=['PC','TPP','VP','SVP']
    vol_props=['VC','LVOL','VDWV','SVR']
    dimless_props=['ZC','ACEN','RI','DC','PAR','SOLW','ACCW', \
                   'FLVL','FLVU']
    energy_props=['HFOR','GFOR','HSTD','GSTD','HFUS','HCOM',\
                  'HSUB','HVP']
    entropy_props=['ENT','SSTD','ICP','LCP','SCP']
    conduct_props=['VTC','LTC','STC']
    visc_props=['LVS','VVS']
    density_props=['LDN','SDN']
    
    if prop in temp_props:
        return(1*u.K)
    elif prop in press_props:
        return(1*u.Pa)
    elif prop in vol_props:
        return(1*u.m**3/u.kmol)
    elif prop in dimless_props:
        return(1)
    elif prop in energy_props:
        return(1*u.J/u.kmol)
    elif prop in entropy_props:
        return(1*u.J/u.kmol/u.K)
    elif prop in conduct_props:
        return(1*u.W/u.m**2/u.K)
    elif prop in visc_props:
        return(1*u.Pa*u.s)
    elif prop in density_props:
        return(1*u.kmol/u.m**3)
    elif prop == 'VDWA':
        return(1*u.m**2/u.kmol)
    elif prop=='RG':
        return(1*u.m)
    elif prop=='SOLP':
        return(1*u.J**0.5/u.m**(1.5))
    elif prop=='DM':
        return(1*u.C*u.m)
    elif prop=='HLC':
        return(1*u.kPa)
    elif prop=='ST':
        return(1*u.N/u.m)
    elif prop=='MW':
        return(1*u.g/u.mol)
    else:
        print("ERROR: Property " + prop + " not found to assign unit.")

def units_are_K(x):
    try:
        if((len(x._unit.keys())==1) and (x._unit['K']==1)):
            return(True)
        else:
            return(False)
    except:
        return(False)
        
def isnumber(s):
    try:
        float(s)
        return(True)
    except ValueError:
        return(False)

# The class for each tdep property
class tcoeff:
    def __init__(self):
        self.prop=''            # property
        self.tmin=np.nan        # min temp of correlation
        self.tmax=np.nan        # max temp of correlation
        self.eq=np.nan          # correlation eqation number
        self.c=np.array([])     # coefficients
        self.devmin=np.nan      # minimum deviation from correlation
        self.devmax=np.nan      # maximum deviation from correlation
        self.dtype=''           # data type of correlation
        self.error=''           # error of correlation
        self.noteID=''          # note id

# The class for the constant property metadata   
class metadata:
    def __init__(self):
        self.refID=''           # refid
        self.noteID=''          # note ID
        self.error=''           # error of point
        self.dtype=''           # data type of point
        self.sourcedetail=''    # source detail, primary/cited/etc.
        self.sourcetype=''      # source type, evaluated/unevaluated/staff
        self.errorsource=''     # error source
        

        
# The class to hold the compound information
class compound:
    def __init__(self):
        self.Name=''
        self.ChemID=''
        self.Formula=''
        self.CAS=''
        self.MW=np.nan
        self.TC=np.nan
        self.PC=np.nan
        self.VC=np.nan
        self.ZC=np.nan  
        self.MP=np.nan  
        self.TPT=np.nan 
        self.TPP=np.nan 
        self.NBP=np.nan 
        self.LVOL=np.nan
        self.HFOR=np.nan
        self.GFOR=np.nan
        self.ENT=np.nan 
        self.HSTD=np.nan
        self.GSTD=np.nan
        self.SSTD=np.nan
        self.HFUS=np.nan
        self.HCOM=np.nan
        self.ACEN=np.nan
        self.RG=np.nan  
        self.SOLP=np.nan
        self.DM=np.nan  
        self.VDWA=np.nan
        self.VDWV=np.nan
        self.RI=np.nan  
        self.FP=np.nan  
        self.FLVL=np.nan
        self.FLVU=np.nan
        self.FLTL=np.nan
        self.FLTU=np.nan
        self.AIT=np.nan 
        self.HSUB=np.nan
        self.PAR=np.nan 
        self.DC=np.nan  
        self.HLC=np.nan 
        self.SOLW=np.nan
        self.ACCW=np.nan
        cprops=['MW','TC','PC','VC','ZC','MP','TPT','TPP','NBP', \
                'LVOL','HFOR','GFOR','ENT','HSTD','GSTD','SSTD', \
                'HFUS','HCOM','ACEN','RG','SOLP','DM','VDWA','VDWV', \
                'RI','FP','FLVL','FLVU','FLTL','FLTU','AIT','HSUB', \
                'PAR','DC','HLC','SOLW','ACCW']
        self.mdata={}
        for i in cprops:
            self.mdata[i]=metadata()
        tprops=['LDN','SDN','ICP','LCP','SCP','HVP','SVR','ST', \
                'LTC','VTC','STC','VP','SVP','LVS','VVS']
        self.coeff={}
        for i in tprops:
            self.coeff[i]=tcoeff()
                
    def LDN(self,t):
        """liquid density of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the liquid density of the compound at temperature `t` in kmol/m**3
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['LDN'].eq):
            print("ERROR: LDN has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false)
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['LDN'].eq == 116 or self.coeff['LDN'].eq == 119: T = 1-T/TC
            return(eq.eq(T,self.coeff['LDN'].c,self.coeff['LDN'].eq)*add_dippr_units('LDN'))
        else:
            print("ERROR: LDN requires unum units of 'K'.")
            return(np.nan)
    
    def SDN(self,t):
        """solid density of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the solid density of the compound at temperature `t` in kmol/m**3
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['SDN'].eq):
            print("ERROR: SDN has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false)
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['SDN'].c,self.coeff['SDN'].eq)*add_dippr_units('SDN'))
        else:
            print("ERROR: SDN requires unum units of 'K'.")
            return(np.nan)
        
    
    def ICP(self,t):
        """ideal gas heat capacity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the ideal gas heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['ICP'].eq):
            print("ERROR: ICP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['ICP'].c,self.coeff['ICP'].eq)*add_dippr_units('ICP'))
        else:
            print("ERROR: ICP requires unum units of 'K'.")
            return(np.nan)
    
    def LCP(self,t):
        """liquid heat capacity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the liquid heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['LCP'].eq):
            print("ERROR: LCP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['LCP'].eq == 114 or self.coeff['LCP'].eq == 124: T = 1-T/TC
            return(eq.eq(T,self.coeff['LCP'].c,self.coeff['LCP'].eq)*add_dippr_units('LCP'))
        else:
            print("ERROR: LCP requires unum units of 'K'.")
            return(np.nan)
    
    def SCP(self,t):
        """solid heat capacity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the solid heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['SCP'].eq):
            print("ERROR: SCP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['SCP'].c,self.coeff['SCP'].eq)*add_dippr_units('SCP'))
        else:
            print("ERROR: SCP requires unum units of 'K'.")
            return(np.nan)
    
    def HVP(self,t):
        """heat of vaporization of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the heat of vaporization of the compound at temperature `t` in J/kmol
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['HVP'].eq):
            print("ERROR: HVP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['HVP'].eq == 106: T = T/TC
            return(eq.eq(T,self.coeff['HVP'].c,self.coeff['HVP'].eq)*add_dippr_units('HVP'))
        else:
            print("ERROR: HVP requires unum units of 'K'.")
            return(np.nan)
    
    def SVR(self,t):
        """second virial coefficient of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the second virial coefficient of the compound at temperature `t` in m**3/kmol
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['SVR'].eq):
            print("ERROR: SVR has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['SVR'].c,self.coeff['SVR'].eq)*add_dippr_units('SVR'))
        else:
            print("ERROR: SVR requires unum units of 'K'.")
            return(np.nan)

    
    def ST(self,t):
        """ surface tension of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the surface tension of the compound at temperature `t` in N/m
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['ST'].eq):
            print("ERROR: ST has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['ST'].eq == 106: T = T/TC
            return(eq.eq(T,self.coeff['ST'].c,self.coeff['ST'].eq)*add_dippr_units('ST'))
        else:
            print("ERROR: ST requires unum units of 'K'.")
            return(np.nan)
    
    def LTC(self,t):
        """ liquid thermal conductivity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the liquid thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['LTC'].eq):
            print("ERROR: LTC has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['LTC'].eq == 123: T = 1-T/TC
            return(eq.eq(T,self.coeff['LTC'].c,self.coeff['LTC'].eq)*add_dippr_units('LTC'))
        else:
            print("ERROR: LTC requires unum units of 'K'.")
            return(np.nan)
    
    def VTC(self,t):
        """ vapor thermal conductivity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the vapor thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['VTC'].eq):
            print("ERROR: VTC has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['VTC'].c,self.coeff['VTC'].eq)*add_dippr_units('VTC'))
        else:
            print("ERROR: VTC requires unum units of 'K'.")
            return(np.nan)
    
    def STC(self,t):
        """ solid thermal conductivity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the solid thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        if np.isnan(self.coeff['STC'].eq):
            print("ERROR: STC has not been initialized for this compound.")
            return(np.nan)
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['STC'].c,self.coeff['STC'].eq)*add_dippr_units('STC'))
        else:
            print("ERROR: STC requires unum units of 'K'.")
            return(np.nan)
    
    def VP(self,t):
        """ liquid vapor pressure of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the saturated vapor pressure of the compound at temperature `t` in Pa
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['VP'].eq):
            print("ERROR: VP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['VP'].c,self.coeff['VP'].eq)*add_dippr_units('VP'))
        else:
            print("ERROR: VP requires unum units of 'K'.")
            return(np.nan)
    
    def SVP(self,t):
        """ solid vapor pressure of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the pressure of the vapor in equilibrium with the solid of the compound at temperature `t` in Pa
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['SVP'].eq):
            print("ERROR: SVP has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['SVP'].c,self.coeff['SVP'].eq)*add_dippr_units('SVP'))
        else:
            print("ERROR: SVP requires unum units of 'K'.")
            return(np.nan)

    
    def LVS(self,t):
        """ liquid viscosity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the liquid viscosity of the compound at temperature `t` in Pa*s
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['LVS'].eq):
            print("ERROR: LVS has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['LVS'].c,self.coeff['LVS'].eq)*add_dippr_units('LVS'))
        else:
            print("ERROR: LVS requires unum units of 'K'.")
            return(np.nan)
    
    def VVS(self,t):
        """ vapor viscosity of the compound
        
        Parameter
        ----------
        t : unum
            temperature (K)
            
        Returns
        -------
        unum
            the low-pressure vapor viscosity of the compound at temperature `t` in Pa*s
        """
        # Check to see if the property has been initialized; return error if not
        if np.isnan(self.coeff['VVS'].eq):
            print("ERROR: VVS has not been initialized for this compound.")
            return(np.nan)
        # Check that the units on `t` are K; return property (true) or error (false) 
        if units_are_K(t):
            T=t.asNumber()
            return(eq.eq(T,self.coeff['VVS'].c,self.coeff['VVS'].eq)*add_dippr_units('VVS'))
        else:
            print("ERROR: VVS requires unum units of 'K'.")
            return(np.nan)
        
 
 