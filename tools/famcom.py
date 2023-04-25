# famcom is used to graph property data from a list of DIPPR compounds      #
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
# famcom.py                                                                 #
#                                                                           #
# Thomas A. Knotts IV                                                       #
# Brigham Young University                                                  #
# Department of Chemical Engineering                                        #
# Provo, UT  84606                                                          #
# Email thomas.knotts@byu.edu                                               #
# ========================================================================= #
# Version 1.0 - September 2022                                              #
# ========================================================================= #

import math
import matplotlib.pyplot as plt
import numpy as np
import byutpl.tools.fitting as fit
def graphs(c,p):
    """Graphs the data for the compounds in `c` for property `p`
    
    
    
    Parameters
    ----------
    c : list of `byutpl.tools.compound` objects
        
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
            for i in range(len(cindex)):
                xdata=c[cindex[i]].MW
                ydata=getattr(c[cindex[i]],p)
                plt.plot(xdata,ydata,'o',label=c[cindex[i]].Name)
            plt.legend(loc=(1.04, 0))
            plt.ylabel(p)
            plt.xlabel('MW')
            plt.title(p + ' vs MW')
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


def othmer(c):
    """Graphs the Othmer plots for VP
    
    
    
    Parameters
    ----------
    c : list of `byutpl.tools.compound` objects
     
    Graphs the Othmer plots for all compounds in `c`. The VP's are
    normalized with respect to the compound with the lowest NBP.
    """
    # check if `c` is a list
    if type(c) != list:
        print('Graphing requires a list of compound objects which was not supplied.')
        return()
   
    # sort c on NBP
    nbpindex=[i for i, x in enumerate(c) if not math.isnan(x.NBP)]
    if(bool(nbpindex)): c.sort(key=lambda x: x.NBP)   
    

    # Determine the index of the compounds in `c` have data for property `p`
    cindex=[i for i, x in enumerate(c) if not math.isnan(x.coeff['VP'].eq)]
 
    if not cindex: # only graph if data are present
        print('No data for VP were found in the supplied files, so the Othmer Plot ' + \
              'can\'t be created.') 
        return()
    else:
        #determine min and max to plot
        xmin=np.max(np.array([i.coeff['VP'].tmin for i in c]))
        xmax=np.min(np.array([i.coeff['VP'].tmax for i in c]))
        xdata=np.linspace(xmin,xmax,100)
        ydata=np.zeros([len(c),len(xdata)])
        ydata0=c[cindex[0]].VP(xdata)
        for i in range(len(cindex)):
            ydata=c[cindex[i]].VP(xdata)/ydata0
            plt.plot(xdata,ydata,label=c[cindex[i]].Name)
            plt.yscale('log')

        plt.legend(loc=(1.04, 0))
        plt.ylabel('VP Ratio')
        plt.xlabel('T (K)')
        plt.title('Othmer Plot')
        plt.show()

def troutontau(c):
    """Determines and graphs the tau0 Trouton parameters
    
    Parameters
    ----------
    c : list of `byutpl.tools.compound` objects
     
    Determines the Trouton parameters (tau0) for the compounds in `c`
    based on their vapor pressures and then graphs them by molecular
    weight.
    """
    # sort c on MW if MW is available
    mwindex=[i for i, x in enumerate(c) if not math.isnan(x.MW)]
    if(bool(mwindex)): c.sort(key=lambda x: x.MW)

    # Determine the index of the compounds in `c` have data for property `VP`
    cindex=[i for i, x in enumerate(c) if not math.isnan(x.coeff['VP'].eq)]
 
    if not cindex: # only graph if data are present
        print('No data for VP were found in the supplied files, so the Trouton parameter ' + \
              'plot can\'t be created.') 
        return()
    else:
        for i in range(len(cindex)):
            x=np.linspace(c[cindex[i]].coeff['VP'].tmin, c[cindex[i]].coeff['VP'].tmax-1, 50)
            y=c[cindex[i]].VP(x)
            xdata=c[cindex[i]].MW
            param=fit.troutonvp(x,y,c[cindex[i]].NBP,tau1=False,percent=True)
            ydata=param[0]
            plt.plot(xdata,ydata,'o',label=c[cindex[i]].Name)
        plt.legend(loc=(1.04, 0))
        plt.ylabel('Tau0 (kJ/mol)')
        plt.xlabel('MW')
        plt.title('Trouton Parameters')
        plt.show()