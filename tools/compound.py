# compound is used to store and retrieve DIPPR compound data                #
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
# compound.py                                                               #
#                                                                           #
# Thomas A. Knotts IV                                                       #
# Brigham Young University                                                  #
# Department of Chemical Engineering                                        #
# Provo, UT  84606                                                          #
# Email thomas.knotts@byu.edu                                               #
# ========================================================================= #
# Version 1.0 - September 2022                                              #
# Version 1.1 - May 2023 Added metadata and environmental properties        #
#                        Removed `graphs` function to famcom.py module      #
# ========================================================================= #
"""
This module containes the classes to store and retrieve the DIPPR data for 
a compound.

======================================================================
Classes
======================================================================
compound       holds the property information for each compound
tcoeff         holds the information for a temperature-dependent
               property; not usually called directly by user

----------------------------------------------------------------------
Key Method of `compound` Class
----------------------------------------------------------------------
read_compound(fn)
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
import byutpl.equations.dippreqns as eq

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

    def read_compound(self,fn):
        """reads the property data into the self `compound` object
        
        Parameter
        ----------
        fn : string
             name of file containing the data for the compound in 
             'key\tvalue(s)' form
            
        """
        # check to see if the input files exists
        if not os.path.isfile(fn): 
            print("Input file \"" + fn +"\" does not exist.\n")
            import warnings
            warnings.filterwarnings("ignore")
            sys.exit("Error: Input file missing.")
        
        # Parse the input file
        fi=open(fn)        # open the file
        content=fi.readlines()  # read the file into a variable
        fi.close()              # close the file
        data={}               # make a dicitonary to hold the keywords and values
        for line in content:          # interate through each line of text in file
            linetext=line.strip()     # get rid of whitespace on each end of line
            linetext=linetext.rstrip('\t') # remove triling tabs
            if not linetext: continue # skip empty lines
            # Remove the end of line comment (everything after '#') and
            # and split the lines at all commas
            linetext=linetext.split('#',1)[0].split('\t')
            if not linetext: continue # skip a line that was only comments
            # Separate the line into the key and the value pair
            if len(linetext) > 1: # ignore keys without values
                key=linetext[0]            
                val=linetext[1:]
                # Place the key and value into the dictionary
                data[key]=val
        # Assign constant data
        cprops=['MW','TC','PC','VC','ZC','MP','TPT','TPP','NBP','LVOL','HFOR','GFOR', \
                'ENT','HSTD','GSTD','SSTD','HFUS','HCOM','ACEN','RG','SOLP','DM', \
                'VDWA','VDWV','RI','FP','FLVL','FLTL','FLVU','FLTU','AIT','HSUB', \
                'PAR','DC']
        if 'Name' in data.keys(): self.Name=data.get('Name')[0]
        if 'ChemID' in data.keys(): self.ChemID=int(data.get('ChemID')[0])
        for i in cprops:
            if i in data: # check if prop was in file
                setattr(self, i, float(data.get(i)[0]))
        
        # tdep coefficients
        tprops=['LDN','SDN','ICP','LCP','SCP','HVP','SVR','ST', \
                'LTC','VTC','STC','VP','SVP','LVS','VVS']
        for i in tprops:
            if i in data: # check if prop was in file
                self.coeff[i].eq=int(data.get(i)[0])
                self.coeff[i].tmin=float(data.get(i)[1])
                self.coeff[i].tmax=float(data.get(i)[2])
                self.coeff[i].c=np.array(data.get(i)[3:]).astype(float)
                
    def LDN(self,t):
        """liquid density of the compound
        
        Parameter
        ----------
        t : float
            temperature (K)
            
        Returns
        -------
        float
            the liquid density of the compound at temperature `t` in kmol/m**3
        """
        if self.coeff['LDN'].eq == 116 or self.coeff['LDN'].eq == 119: t = 1-t/self.TC
        return(eq.eq(t,self.coeff['LDN'].c,self.coeff['LDN'].eq))
    
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

 