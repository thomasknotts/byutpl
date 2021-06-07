#byutpl

byutpl is a suite of tools for chemical and other engineers to
to 1) perform thermodynamic calculations using cubic equations of
state and 2) obtain thermophysical properties for a limited set
of compunds.

Its target audience is students in chemical and mechanical engineering
majors. Before using this tool, students should fully understand the 
principles behind the calculations being performed.

#Usage
import byutpl.eos.srk as srk
import byutpl.eos.pr as pr
import byutpl.properties.water as wtr
import byutpl.properties.air as air
import byutpl.properties.benzene as bzn

The following command returns the residual heat capacity calculated by 
the Soave, Redlich, Kwong equation of state at 300 K
and 5E5 Pa for the liquid phase of the compound described by 
critical temperature = 369 K
critical pressure = 480000 Pa
acentric factor = 0.81


srk.hrl(300,5E5,369,48E5,0.81)


The following command returns the liquid viscosity of water at 400 K.
wtr.lvs(400)


For complete lists of functions, properties, and units see
help(srk)
help(pr)
help(wtr)
help(air)
help(bzn)

#Developer
Thomas A. Knotts
Brigham Young University Thermophysical Properties Laboratory

#License
GPL(https://www.gnu.org/licenses/gpl-3.0.txt)
 
