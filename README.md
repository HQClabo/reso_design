# Resonator design
Package to help designing resonators

## Installation
Clone this repository on your computer, navigate to the top folder of the repository and execute from the command line:
`pip install -e .`

This will install the package with "edit" mode, i.e. if you change the source codeor pull changes from origin, the package installed on you rPC will always be up to date automatically.

## Example usage

### The `JJ_array` class
The most easy way to design a resonator is to cleate a `JJ_array` object and when print it. This is how it looks:

```
from reso_design import Junction, JJ_array

# Decide the physical parameters of the JJ array
d_bottom = 20e-9 / np.sqrt(2) # metal thickness
d_top = 40e-9 / np.sqrt(2) # metal thickness
t = 1e-9        # oxide thickness
w = 0.515e-6      # junction width
l_j = 0.5e-6    # junction length
l_s = 3.0e-6    # spurious junction length
l_u = 4.5e-6    # unit cell (1 bridge length + 1 bridge to bridge separation)
s = 40e-6 - w/2 # spacing from ground plane
H = 525e-6      # substrate thickness
RA = 650e-12    # junction's normalized RT resistance in Ohm m^2
N = 30          # number of junctions

# Create the Junction object and then the JJ_array object
junction = Junction(d_top, d_bottom, t, w)
reso = JJ_array(junction, w, s, H, l_j, l_s, l_u, N, RA)

# Print a summary of all the important quantities of the resonator
print(reso)
```