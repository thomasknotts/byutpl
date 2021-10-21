# byutpl

byutpl (Brigham Young University Thermophysical Properties Laboratory)
is a suite of tools for chemical and other engineers to
1. perform thermodynamic calculations using cubic equations of
state and
2. obtain thermophysical properties for a limited set
of compunds.

Its target audience is students in chemical and mechanical engineering
majors. Before using this tool, students should fully understand the 
principles behind the calculations being performed.

It is most useful for courses in **thermodynamics**, **fluids**, and
**heat and mass transfer**.

# Dependencies
- numpy
- scipy

# Usage
## Available Packages
### Import the module for the Soave, Redlich, Kwong equation of state
```import byutpl.eos.srk as srk```
### Import the module for the Peng-Robinson equation of state
```import byutpl.eos.pr as pr```
### Import the modules for the properties of water
```import byutpl.properties.water as wtr```
### Import the package for the properties of air
```import byutpl.properties.air as air```
### Import the package for the properties of benzene
```import byutpl.properties.benzene as bzn```

## Example Commands
The following command returns the residual heat capacity in units of J/mol,
calculated using the Soave, Redlich, Kwong equation of state, at 300 K
and 500000 Pa for the liquid phase of the compound described by 
- critical temperature = 369 K 
- critical pressure = 4800000 Pa 
- acentric factor = 0.81

```srk.hrl(300,5E5,369,48E5,0.81)```

The following command returns the liquid viscosity of water at 400 K in units of Pa*s

```wtr.lvs(400)```

## Complete Documentation
For complete lists of functions, properties, and units use the 
```help()``` function after importing the respective module. e.g.
```help(srk)```
```help(pr)```
```help(wtr)```
```help(air)```
```help(bzn)```

# Developer
Thomas A. Knotts

Brigham Young University Thermophysical Properties Laboratory

knotts.byu.edu(http://knotts.byu.edu/)

# License
GPL(https://www.gnu.org/licenses/gpl-3.0.txt)
 
