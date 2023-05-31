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
        self.tmin=float("nan")  # min temp of correlation
        self.tmax=float("nan")  # max temp of correlation
        self.eq=float("nan")    # correlation eqation number
        self.c=np.array([])     # coefficients
        self.devmin=float("nan")# minimum deviation from correlation
        self.devmax=float("nan")# maximum deviation from correlation
        self.dtype=''           # data type of correlation
        self.error=''           # error of correlation
        self.noteID=-1          # note id

# The class for the constant property metadata   
class metadata:
    def __init__(self):
        self.refID=-1           # refid
        self.noteID=-1          # note ID
        self.error=''           # error of point
        self.dtype=''           # data type of point

        
# The class to hold the compound information
class compound:
    def __init__(self):
        self.Name=''
        self.ChemID=-1
        self.MW=float("nan")
        self.TC=float("nan")
        self.PC=float("nan")
        self.VC=float("nan")
        self.ZC=float("nan")
        self.MP=float("nan")
        self.TPT=float("nan")
        self.TPP=float("nan")
        self.NBP=float("nan")
        self.LVOL=float("nan")
        self.HFOR=float("nan")
        self.GFOR=float("nan")
        self.ENT=float("nan")
        self.HSTD=float("nan")
        self.GSTD=float("nan")
        self.SSTD=float("nan")
        self.HFUS=float("nan")
        self.HCOM=float("nan")
        self.ACEN=float("nan")
        self.RG=float("nan")
        self.SOLP=float("nan")
        self.DM=float("nan")
        self.VDWA=float("nan")
        self.VDWV=float("nan")
        self.RI=float("nan")
        self.FP=float("nan")
        self.FLVL=float("nan")
        self.FLVU=float("nan")
        self.FLTL=float("nan")
        self.FLTU=float("nan")
        self.AIT=float("nan")
        self.HSUB=float("nan")
        self.PAR=float("nan")
        self.DC=float("nan")
        self.HLC=float("nan")
        self.SOLW=float("nan")
        self.ACCW=float("nan")
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
        float
            the liquid density of the compound at temperature `t` in kmol/m**3
        """
        if units_are_K(t):
            T=t.asNumber()
            if units_are_K(self.TC):
                TC=self.TC.asNumber()
            else:
                TC=self.TC
            if self.coeff['LDN'].eq == 116 or self.coeff['LDN'].eq == 119: T = 1-T/TC
            return(eq.eq(T,self.coeff['LDN'].c,self.coeff['LDN'].eq)*add_dippr_units('LDN'))
        else:
            print("LDN requires unum units of 'K'.")
            return(np.nan)
    
    def SDN(self,t):
        """solid density of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the solid density of the compound at temperature `t` in kmol/m**3
        """
        return(eq.eq(t,self.coeff['SDN'].c,self.coeff['SDN'].eq))
    
    def ICP(self,t):
        """ideal gas heat capacity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the ideal gas heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        return(eq.eq(t,self.coeff['ICP'].c,self.coeff['ICP'].eq))
    
    def LCP(self,t):
        """liquid heat capacity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the liquid heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        if self.coeff['LCP'].eq == 114 or self.coeff['LCP'].eq == 124: t = 1-t/self.TC
        return(eq.eq(t,self.coeff['LCP'].c,self.coeff['LCP'].eq))
    
    def SCP(self,t):
        """solid heat capacity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the solid heat capacity of the compound at temperature `t` in J/(kmol*K)
        """
        return(eq.eq(t,self.coeff['SCP'].c,self.coeff['SCP'].eq))
    
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
        if self.coeff['HVP'].eq == 106: t = t/self.TC
        return(eq.eq(t,self.coeff['HVP'].c,self.coeff['HVP'].eq))
    
    def SVR(self,t):
        """second virial coefficient of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the second virial coefficient of the compound at temperature `t` in m**3/kmol
        """
        return(eq.eq(t,self.coeff['SVR'].c,self.coeff['SVR'].eq))
    
    def ST(self,t):
        """ surface tension of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the surface tension of the compound at temperature `t` in N/m
        """
        if self.coeff['ST'].eq == 106: t = t/self.TC
        return(eq.eq(t,self.coeff['ST'].c,self.coeff['ST'].eq))
    
    def LTC(self,t):
        """ liquid thermal conductivity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the liquid thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        if self.coeff['LTC'].eq == 123: t = 1-t/self.TC
        return(eq.eq(t,self.coeff['LTC'].c,self.coeff['LTC'].eq))
    
    def VTC(self,t):
        """ vapor thermal conductivity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the vapor thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        return(eq.eq(t,self.coeff['VTC'].c,self.coeff['VTC'].eq))
    
    def STC(self,t):
        """ solid thermal conductivity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the solid thermal conductivity of the compound at temperature `t` in W/(m*K)
        """
        return(eq.eq(t,self.coeff['STC'].c,self.coeff['STC'].eq))
    
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
        return(eq.eq(t,self.coeff['VP'].c,self.coeff['VP'].eq))
    
    def SVP(self,t):
        """ solid vapor pressure of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the pressure of the vapor in equilibrium with the solid of the compound at temperature `t` in Pa
        """
        return(eq.eq(t,self.coeff['SVP'].c,self.coeff['SVP'].eq))
    
    def LVS(self,t):
        """ liquid viscosity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the liquid viscosity of the compound at temperature `t` in Pa*s
        """
        return(eq.eq(t,self.coeff['LVS'].c,self.coeff['LVS'].eq))
    
    def VVS(self,t):
        """ vapor viscosity of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the low-pressure vapor viscosity of the compound at temperature `t` in Pa*s
        """
        return(eq.eq(t,self.coeff['VVS'].c,self.coeff['VVS'].eq))
        
def graphs(c,p):
    """Graphs the data for the compounds in `c` for property `p`
    
    
    
    Parameters
    ----------
    c : list of `famcom.compound` objects
        
    p : string
        DIPPR property to graph
        Must be one of the following: MW, TC, PC, VC, ZC, MP, TPT, TPP,
        NBP, LVOL, HFOR, GFOR, ENT, HSTD, GSTD, SSTD, HFUS, HCOM, ACEN,
        RG, SOLP, DM, VDWA, VDWV, RI, FP, FLVL, FLTL, FLVU, FLTU, AIT,
        HSUB, PAR, DC, LDN, SDN, ICP, LCP, SCP, HVP, SVR, ST, LTC, VTC,
        STC, VP, SVP, LVS, VVS.
    
    Graphs property `p` for all compounds in `c`. If `p` is a constant
    property, the graph is done vs molecular weight (`p` vs MW). If `p`
    is a temperature-dependent property, the graph is `p` vs `T`.
    Different lines will appear on the graph if multiple compounds
    are found in `c`.
    """
    # check if `c` is a list
    if type(c) != list:
        print('Graphing requires a list of compound objects which was not supplied.')
        return()
   
    # check whether the property is constant, tdep, or not a DIPPR prop
    cprops=['MW','TC','PC','VC','ZC','MP','TPT','TPP','NBP','LVOL','HFOR','GFOR', \
            'ENT','HSTD','GSTD','SSTD','HFUS','HCOM','ACEN','RG','SOLP','DM', \
            'VDWA','VDWV','RI','FP','FLVL','FLTL','FLVU','FLTU','AIT','HSUB', \
            'PAR','DC']
    tprops=['LDN','SDN','ICP','LCP','SCP','HVP','SVR','ST', \
            'LTC','VTC','STC','VP','SVP','LVS','VVS']
    ptype=''
    if p in cprops: ptype='const'
    elif p not in tprops:
        print('Property ' + p + ' is not a DIPPR property.')
        return()
    
    # sort c on MW if MW is available
    mwindex=[i for i, x in enumerate(c) if not math.isnan(x.MW)]
    if(bool(mwindex)): c.sort(key=lambda x: x.MW)
    
    # Determine the index of the compounds in `c` have data for property `p`
    if ptype == 'const': cindex=[i for i, x in enumerate(c) if not math.isnan(getattr(x,p))]
    else: cindex=[i for i, x in enumerate(c) if not math.isnan(x.coeff[p].eq)]
 
    if not cindex: # only graph if data are present
        print('No data for ' + p + ' were found in the supplied files.') 
        return()
    else:
        if ptype == 'const':
            names=[c[i].name for i in cindex]
            xdata=[c[i].MW for i in cindex]
            ydata=[getattr(c[i],p) for i in cindex]
            plt.plot(xdata,ydata,'o')
            plt.ylabel(p)
            plt.xlabel('MW')
            plt.title(p + ' vs MW')
            print(names)
        else:
            for i in range(len(cindex)):
                xdata=np.linspace(c[cindex[i]].coeff[p].tmin, c[cindex[i]].coeff[p].tmax-1, 50)
                yf=getattr(c[cindex[i]],p)
                ydata=yf(xdata)
                if p in ['VP','SVP','LVS']:
                    xdata=1.0/xdata
                    ydata=np.log(ydata)
                plt.plot(xdata,ydata,label=c[cindex[i]].Name)
            if p in ['VP','SVP','LVS']:
                plt.ylabel('ln(' + p +')')
                plt.xlabel('1/T')
            else:
                plt.ylabel(p)
                plt.xlabel('T')
            plt.title('Temperature Behavior of ' + p)
            plt.legend(loc=(1.04, 0))
        plt.show()
 
 